from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from transformers import AutoConfig

from lovrr.model.builder import build_model


ROLES = ("planner", "grounder", "refiner", "deepthinker")


def load_role_adapter(model, model_path, role):
    if role in getattr(model, "peft_config", {}):
        return role
    path = Path(model_path)
    candidates = [path / role, path]
    if role == "deepthinker":
        candidates.insert(1, path / "answerer")
    adapter_path = next(
        (candidate for candidate in candidates if (candidate / "adapter_config.json").is_file()),
        None,
    )
    if adapter_path is None:
        raise FileNotFoundError(f"{role} adapter not found in {model_path}")
    model.load_adapter(str(adapter_path), adapter_name=role)
    return role


class RoleModel:
    def __init__(self, model, processor):
        self.model = model
        self.model.eval()
        self.processor = processor
        self.lock = RLock()

    @classmethod
    def from_pretrained(cls, model_path, role_paths=None, device="auto", required_roles=ROLES,
                        base_model_path=None):
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type not in ("qwen2_vl", "agent_qwen2_vl"):
            raise ValueError("model_path must point to a Qwen2-VL checkpoint")
        if base_model_path is not None:
            config.base_model_path = base_model_path
        model, processor = build_model(model_path, config=config, device=device)
        roles = cls(model, processor)
        for role in required_roles:
            roles.load_role((role_paths or {}).get(role) or model_path, role)
        return roles

    @property
    def device(self):
        return next(self.model.parameters()).device

    def load_role(self, model_path, role):
        with self.lock:
            return load_role_adapter(self.model, model_path, role)

    @contextmanager
    def activate(self, role):
        with self.lock:
            if role not in getattr(self.model, "peft_config", {}):
                raise ValueError(f"{role} adapter is not loaded")
            previous = getattr(self.model, "active_adapter", None)
            self.model.base_model.enable_adapter_layers()
            self.model.set_adapter(role)
            try:
                yield self.model
            finally:
                if previous:
                    self.model.set_adapter(previous)

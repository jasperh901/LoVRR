# Copyright (c) 2025. Licensed under the BSD-3-Clause License.

import nncore

from lovrr.dataset.hybrid import DATASETS
from lovrr.dataset.wrappers import GroundingDataset
from lovrr.utils.parser import parse_query


@DATASETS.register(name='ego4d_nlq')
class Ego4DNLQDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/ego4d_nlq/nlq_train.jsonl'
    ANNO_PATH_VALID = 'data/ego4d_nlq/nlq_val.jsonl'

    VIDEO_ROOT = 'data/ego4d/v2/videos_3fps_480_noaudio'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_VALID)

        annos = []
        for raw_anno in raw_annos:
            assert len(raw_anno['relevant_windows']) == 1

            anno = dict(
                source='ego4d_nlq',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, raw_anno['vid'] + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno['relevant_windows'])

            annos.append(anno)

        return annos

# Copyright (c) 2025. Licensed under the BSD-3-Clause License.

import nncore

from lovrr.dataset.hybrid import DATASETS
from lovrr.dataset.wrappers import GroundingDataset
from lovrr.utils.parser import parse_query


@DATASETS.register(name='vid_morp')
class VidMorpDataset(GroundingDataset):

    ANNO_PATH = 'data/vid_morp/anno_vid_morp.jsonl'

    VIDEO_ROOT = 'data/vid_morp/videos_3fps_480_noaudio'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH)

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['vid']

            if vid not in all_videos:
                continue

            anno = dict(
                source='vid_morp',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=[raw_anno['span']])

            annos.append(anno)

        return annos

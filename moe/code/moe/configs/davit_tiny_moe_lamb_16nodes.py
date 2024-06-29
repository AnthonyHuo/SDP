# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]

# dataset settings
cls_dataset_type = 'ImageNet'
cls_data_root = 'data/ILSVRC2012/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

cls_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

cls_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeCls',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# dataset settings
det_dataset_type = 'CocoDataset'
det_data_root = 'data/COCO/'
det_train_pipeline = [
    dict(type='LoadImageFromFileDet'),
    dict(type='LoadAnnotationsDet', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlipDet', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadDet', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectDet', keys=['img', 'gt_bboxes', 'gt_labels']),
]
det_test_pipeline = [
    dict(type='LoadImageFromFileDet'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlipDet'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='PadDet', size_divisor=32),
            dict(type='ImageToTensorDet', keys=['img']),
            dict(type='CollectDet', keys=['img']),
        ])
]

# dataset settings
seg_dataset_type = 'ADE20KDataset'
seg_data_root = 'data/ADEChallengeData2016/'
crop_size = (512, 512)
seg_train_pipeline = [
    dict(type='LoadImageFromFileSeg'),
    dict(type='LoadAnnotationsSeg', reduce_zero_label=True),
    dict(type='ResizeSeg', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCropSeg', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipSeg', prob=0.5),
    dict(type='PhotoMetricDistortionSeg'),
    dict(type='NormalizeSeg', **img_norm_cfg),
    dict(type='PadSeg', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundleSeg'),
    dict(type='CollectSeg', keys=['img', 'gt_semantic_seg']),
]
seg_test_pipeline = [
    dict(type='LoadImageFromFileSeg'),
    dict(
        type='MultiScaleFlipAugSeg',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='ResizeSeg', keep_ratio=True),
            dict(type='RandomFlipSeg'),
            dict(type='NormalizeSeg', **img_norm_cfg),
            dict(type='ImageToTensorSeg', keys=['img']),
            dict(type='CollectSeg', keys=['img']),
        ])
]

cat_batchsizes = [64,2,2]  # cls, det, seg
cat_weights = [15, 10, 5]
cat_epochsize = 4000
cat_shuffle = True
cat_dataloader = dict(samples_per_gpu=1, workers_per_gpu=2,
                      sampler_cfg=dict(
                          type='Distributed_Weighted_BatchSampler',
                          batch_sizes = cat_batchsizes,
                          weights = cat_weights,
                          num_samples = cat_epochsize,
                          shuffle = cat_shuffle
                      )
                      )

data = dict(
    samples_per_gpu=1,  # must have, otherwise bug report
    workers_per_gpu=1,
    train=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix=cls_data_root + 'train/',pipeline=cls_train_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_train2017.json',img_prefix=det_data_root + 'train2017/',pipeline=det_train_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/training',ann_dir='annotations/training',pipeline=seg_train_pipeline),
        ]
    ),
    train_dataloader=cat_dataloader,
    val=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix=cls_data_root + 'val/',pipeline=cls_test_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_val2017.json',img_prefix=det_data_root + 'val2017/',pipeline=det_test_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/validation',ann_dir='annotations/validation',pipeline=seg_test_pipeline),
        ]
    ),
    # val_dataloader=test_dataloader,
    test=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix=cls_data_root + 'val/',pipeline=cls_test_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_val2017.json',img_prefix=det_data_root + 'val2017/',pipeline=det_test_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/validation',ann_dir='annotations/validation',pipeline=seg_test_pipeline),
        ]
    ),
    # test_dataloader=test_dataloader
)



# _base_ = [
#     # '../configs/_base_/models/retinanet_r50_fpn.py',
#     # '../configs/_base_/datasets/imagenet_coco_ade20k_swin.py',
#     # '../configs/_base_/schedules/imgnet_ade20k_coco_SGD.py',
#     # '../configs/_base_/default_runtime.py'
#     'dataset_config.py',
# ]

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# 5e-4 for 512 = 6e-5 on cls 0.01 for 16 gpu on det 0.00006 for Adam on Seg
optimizer = dict(type='Lamb', lr=1, weight_decay=0.05)  # lr useless here when onecycle used
optimizer_config = dict(grad_clip=dict(max_norm=0.1))
runner = dict(type='EpochBasedRunner', max_epochs=60)
lr_config = dict(policy='OneCycle', max_lr=0.004, total_steps=4000*60)
# momentum_config = dict(
#     policy='OneCycle',
# )


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

evaluation = dict(interval=1, metric=['accuracy', 'bbox', 'mIoU'])

find_unused_parameters = True
# model settings
model = dict(
    type='ImageMTLearner',
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint='model_best.pth.tar'),
        type='DaViT_tiny_moe',
        drop_path_rate=0.2,
        task_num=3,
        ffd_heads=4, num_ffd_experts=64, w_MI=0.0005
    ),
    models=[
        dict(
            task_bh=0,
            task='cls',
            type='ImageClassifier',
            # backbone=dict(
            #     type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss',
                    label_smooth_val=0.1,
                    mode='original',
                    num_classes=1000,
                    reduction='mean',
                    loss_weight=1.0),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
                dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
            ])
        ),
        dict(
            task_bh=1,
            task='det',
            type='RetinaNet',
            neck=dict(
                type='FPN',
                in_channels=[96, 192, 384, 768],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_input',
                num_outs=5),
            bbox_head=dict(
                type='RetinaHead',
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        ),
        dict(
            task_bh=2,
            task='seg',
            type='EncoderDecoder',
            pretrained=None,
            decode_head=dict(
                type='UPerHead',
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLossSeg', use_sigmoid=False, loss_weight=1.0)),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLossSeg', use_sigmoid=False, loss_weight=0.4)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        ),

    ],
    loss_weight=[1.0, 0.6, 0.2]
)
# use_FSDP=True
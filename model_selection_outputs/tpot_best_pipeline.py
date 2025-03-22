# TPOT Best Pipeline
# Score: 0.8520
# Runtime: 3881.59 seconds

best_pipeline = Pipeline(steps=[('robustscaler',
                 RobustScaler(quantile_range=(0.2475690452454,
                                              0.9671736154307))),
                ('selectfwe', SelectFwe(alpha=0.0012903345738)),
                ('featureunion-1',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('featureunion-2',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),...
                               feature_types=None, gamma=0.0153706668713,
                               grow_policy=None, importance_type=None,
                               interaction_constraints=None,
                               learning_rate=0.0354272416933, max_bin=None,
                               max_cat_threshold=None, max_cat_to_onehot=None,
                               max_delta_step=None, max_depth=10,
                               max_leaves=None, min_child_weight=15,
                               missing=nan, monotone_constraints=None,
                               multi_strategy=None, n_estimators=100, n_jobs=1,
                               nthread=1, num_parallel_tree=None, ...))])

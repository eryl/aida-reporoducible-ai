from train import TrainingConfig, HyperParameter

config = TrainingConfig(hidden_dim = HyperParameter(method='suggest_categorical', name='hidden_dim', kwargs={'choices': [128, 256, 512, 1024, 1024]}),
                        dropout_rate = HyperParameter(method='suggest_float', name='dropout_rate', kwargs={'low':0., 'high': 1.}),
                        warmup_learning_rate = HyperParameter(method='suggest_float', name='warmup_learning_rate', kwargs={'low':1e-6, 'high': 1e-2,'log': True}),
                        warmup_weight_decay = HyperParameter(method='suggest_float', name='warmup_weight_decay', kwargs={'low':1e-7, 'high': 1e-5,'log': True}),
                        learning_rate = HyperParameter(method='suggest_float', name='learning_rate', kwargs={'low':1e-7, 'high': 1e-3,'log': True}),
                        weight_decay= HyperParameter(method='suggest_float', name='weight_decay', kwargs={'low':1e-6, 'high': 1e-3, 'log': True}))

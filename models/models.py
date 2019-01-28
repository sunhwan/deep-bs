def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'kdeep':
        from .kdeep_model import KDeepModel
        model = KDeepModel()
    elif opt.model == 'gnina':
        from .gnina_model import GninaModel
        model = GninaModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

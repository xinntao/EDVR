import models.modules.EDVR_arch as EDVR_arch
import models.modules.VSREDUN_arch as VSREDUN_arch
import models.modules.VSRUN_arch as VSRUN_arch


####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])
    elif which_model == 'VSREDUN':
        netG = VSREDUN_arch.VSREDUN(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], res_blocks=opt_net['res_blocks'],
                              res_groups=opt_net['res_groups'], center=opt_net['center'], 
                              w_TSA=opt_net['w_TSA'])
    elif which_model == 'VSRUN':
        netG = VSRUN_arch.VSRUN(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], res_blocks=opt_net['res_blocks'],
                              res_groups=opt_net['res_groups'], center=opt_net['center'], 
                              w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

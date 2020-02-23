import helpers as hp
import options.job_options as jo
import postpro.visualisers.visualiser_ae as va
import postpro.visualisers.visualiser_vae as vv
from postpro.visualisers import visualiser_class as vc


def main():
    class_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='smth1', cut='4q', frames='8', model='tarn_class'))
    ae_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='hmdb1', cut='4q', frames='4', model='i3d_ae'))
    vae_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='hmdb1', cut='4q', frames='4', model='tarn_vae'))

    base_viz = vc.ClassVisualiser(None)
    class_viz = vc.ClassVisualiser(class_spec)
    ae_viz = va.AEVisualiser(ae_spec)
    vae_viz = vv.VAEVisualiser(vae_spec)

    base_viz.plot_temporal_parameter_growth()
    class_viz.plot_class_embeds('train')
    class_viz.plot_class_embeds('train')
    class_viz.plot_frames_target('train', 10)
    class_viz.plot_class_pred('train', 10)
    class_viz.plot_class_loss('train', 10)

    ae_viz.plot_class_target('train', 10)
    ae_viz.plot_class_pred('train', 10)
    ae_viz.plot_class_loss('train', 10)
    ae_viz.plot_frames_target('train', 0, save=True, group=False)
    ae_viz.plot_frames_recon('train', 0, save=True, group=False)
    ae_viz.plot_frames_loss('train', 10)

    vae_viz.plot_class_target('train', 10)
    vae_viz.plot_class_pred('train', 10)
    vae_viz.plot_class_loss('train', 10)
    vae_viz.plot_frames_target('train', 10)
    vae_viz.plot_frames_recon('train', 10)
    vae_viz.plot_frames_loss('train', 10)


if __name__ == '__main__':
    main()

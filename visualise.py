import helpers as hp
import options.job_options as jo
import postpro.visualisers.visualiser_ae as va
from postpro.visualisers import visualiser_class as vc


def main():
    class_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='smth1', cut='4q', frames='4', model='tarn_class'))
    ae_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='smth1', cut='4q', frames='4', model='tarn_ae'))

    base_viz = vc.ClassVisualiser(None)
    class_viz = vc.ClassVisualiser(class_spec)
    ae_viz = va.AEVisualiser(ae_spec)

    base_viz.plot_temporal_parameter_growth()
    class_viz.plot_class_pred('train', 10, save=False)
    class_viz.plot_class_loss('train', 10, save=False)

    ae_viz.plot_class_target('train', 10)
    ae_viz.plot_class_loss('train', 10)
    ae_viz.plot_class_pred('train', 10)
    ae_viz.plot_input('train', 10, save=False, grp=True)
    ae_viz.plot_recon_target('train', 10, save=False, grp=True)
    ae_viz.plot_recon_pred('train', 10, save=False, grp=True)
    ae_viz.plot_recon_loss('train', 10, save=False)


if __name__ == '__main__':
    main()

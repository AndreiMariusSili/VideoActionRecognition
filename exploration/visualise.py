from postpro.visualisers import visualiser_class as vc


def main():
    # class_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='hmdb1', cut='4q', frames='16', model='tarn_class'))
    # ae_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='hmdb1', cut='4q', frames='16', model='tarn_class'))
    # vae_spec = hp.load_spec(jo.VisualiseExperimentOptions(dataset='hmdb1', cut='4q', frames='16', model='tarn_class'))

    base_viz = vc.ClassVisualiser(None)
    # class_viz = vc.ClassVisualiser(class_spec)
    # ae_viz = va.AEVisualiser(ae_spec)
    # vae_viz = vv.VAEVisualiser(vae_spec)

    base_viz.plot_temporal_parameter_growth()
    # class_viz.plot_class_embeds('train')
    # class_viz.plot_tsne()

    # class_viz.plot_class_target('train', 10)
    # class_viz.plot_class_pred('train', 10)
    # class_viz.plot_class_loss('train', 10)
    #
    # ae_viz.plot_class_target('train', 10)
    # ae_viz.plot_class_pred('train', 10)
    # ae_viz.plot_class_loss('train', 10)
    # ae_viz.plot_frames_target('train', 10)
    # ae_viz.plot_frames_recon('train', 10)
    # ae_viz.plot_frames_loss('train', 10)
    #
    # vae_viz.plot_class_target('train', 10)
    # vae_viz.plot_class_pred('train', 10)
    # vae_viz.plot_class_loss('train', 10)
    # vae_viz.plot_frames_target('train', 10)
    # vae_viz.plot_frames_recon('train', 10)
    # vae_viz.plot_frames_loss('train', 10)


if __name__ == '__main__':
    main()

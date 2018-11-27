import options
import utils
from trainer import TrainerStage1

if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain structure generator with fixed viewpoints")
    print("=======================================================")

    cfg = options.get_arguments()

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    MODEL_PATH = f"models/{EXPERIMENT}"
    LOG_PATH = f"logs/{EXPERIMENT}"

    utils.make_folder(MODEL_PATH)
    utils.make_folder(MODEL_PATH)

    criterions = utils.define_losses()
    dataloaders = utils.make_data_fixed(cfg)

    model = utils.build_structure_generator(cfg).to(cfg.device)
    optimizer = utils.make_optimizer(cfg, model)
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    logger = utils.make_logger(LOG_PATH)
    writer = utils.make_summary_writer(EXPERIMENT)
    
    def on_after_epoch(model, df_hist, images, epoch):
        utils.save_best_model(MODEL_PATH, model, df_hist)
        utils.log_hist(logger, df_hist)
        utils.write_on_board_losses_stg1(writer, df_hist)
        utils.write_on_board_images_stg1(writer, images, epoch)

    if cfg.lrSched is not None:
        def on_after_batch(iteration):
            utils.write_on_board_lr(writer, scheduler.get_lr(), iteration)
            scheduler.step(iteration)
    else: on_after_batch = None

    trainer = TrainerStage1(
        cfg, dataloaders, criterions, on_after_epoch, on_after_batch)

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"{LOG_PATH}.csv", index=False)
    writer.close()

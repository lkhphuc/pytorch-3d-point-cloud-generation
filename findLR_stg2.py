import options
import utils
from trainer import TrainerStage2

if __name__ == "__main__":

    print("=======================================================")
    print("Find optimal LR for joint training with novel viewpoints by pseudo-rendere")
    print("=======================================================")

    cfg = options.get_arguments()

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}_findLR"

    criterions = utils.define_losses()
    dataloaders = utils.make_data_novel(cfg)

    model = utils.build_structure_generator(cfg).to(cfg.device)
    optimizer = utils.make_optimizer(cfg, model)

    writer = utils.make_summary_writer(EXPERIMENT)

    trainer = TrainerStage2(cfg, dataloaders, criterions)
    trainer.findLR(model, optimizer, writer,
                   start_lr=cfg.startLR, end_lr=cfg.endLR,
                   num_iters=cfg.itersLR) 

    writer.close()

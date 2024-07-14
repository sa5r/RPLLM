from datetime import datetime
import argparse
import torch
from utils import Utils

def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', required = True, type = float)
    parser.add_argument('--dataset_directory', required = True, type = str)
    parser.add_argument('--entities_filename', required = True, type = str)
    parser.add_argument('--model_id', required = True, type = str)
    parser.add_argument('--repo_token', required = True, type = str)
    parser.add_argument('--padding', required = True, type = int)
    parser.add_argument('--learning_rate', required = True, type = float)
    parser.add_argument('--decay', required = True, type = float)
    parser.add_argument('--task', required = True, type = str)
    parser.add_argument('--batch_size', required = True, type = int)
    parser.add_argument('--patience', required = True, type = int)
    parser.add_argument('--epochs', required = True, type = int)
    parser.add_argument('--verbose', required = True)
    parser.add_argument('--attention_dropout', required = True, type=float)
    parser.add_argument('--checkpoint_path', required = True, type=str)

    args = parser.parse_args()

    # Random seeds
    g = torch.Generator()
    g.manual_seed(0)

    # Initializations
    
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    # To be removed from the online repo

    # Load utilities
    utils = Utils(time_stamp)
    relations = utils.load_relations(args.dataset_directory + '/relations.txt')    

    # Loading GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.write_log('\ndevice ' + str(device))

    train(relations=relations, args=args, \
            utils = utils, generator=g,
            training_triples='train.tsv',
            validation_triples='dev.tsv',
            device=device)
    
    # Load training set
    training_set = KGDataset(args=args,
                             relations=relations,
                             triples_filename=training_triples,
                             is_training=True)
    training_generator = DataLoader(training_set,
                                    batch_size = args.batch_size,
                                    worker_init_fn=utils.seed_worker,
                                    generator=generator,shuffle=True)
    validation_set = KGDataset(args=args,
                               relations=relations,
                               triples_filename=validation_triples)
    validation_generator = DataLoader(validation_set,
                                      batch_size = args.batch_size,
                                    worker_init_fn=utils.seed_worker,
                                    generator=generator, shuffle=True)

if __name__ == "__main__": main()
from datetime import datetime
import argparse
import torch
from tqdm import tqdm
from utils import Utils
from model import KGDataset,Llama
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# Prevents many tokenizer warnings
transformers.logging.set_verbosity_error()

def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required = True, type=str)
    parser.add_argument('--dataset_directory', required = True, type = str)
    parser.add_argument('--repo_token', required = True, type = str)
    parser.add_argument('--data_size', required = True, type = float)
    parser.add_argument('--model_id', required = True, type = str)
    parser.add_argument('--entities_filename', required = True, type = str)
    parser.add_argument('--descriptions_filename', required = True, type = str)
    parser.add_argument('--padding', required = True, type = int)
    parser.add_argument('--batch_size', required = True, type = int)
    parser.add_argument('--patience', required = True, type = int)
    parser.add_argument('--learning_rate', required = True, type = float)
    parser.add_argument('--decay', required = True, type = float)
    parser.add_argument('--epochs', required = True, type = int)
    parser.add_argument('--verbose', required = True)
    parser.add_argument('--attention_dropout', required = True, type=float)

    args = parser.parse_args()

    # Random seeds
    g = torch.Generator()
    g.manual_seed(0)

    # Initializations
    
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    # Load utilities
    utils = Utils(time_stamp)
    relations = utils.load_relations(args.dataset_directory + '/relations.txt')    

    # Loading GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.write_log('\ndevice ' + str(device))
    
    # Load training set
    training_set = KGDataset(args=args,
                             relations=relations,
                             triples_filename='train.tsv',
                             is_training=True)
    training_generator = DataLoader(training_set,
                                    batch_size = args.batch_size,
                                    worker_init_fn=utils.seed_worker,
                                    generator=g,shuffle=True)
    validation_set = KGDataset(args=args,
                               relations=relations,
                               triples_filename='dev.tsv')
    validation_generator = DataLoader(validation_set,
                                      batch_size = args.batch_size,
                                    worker_init_fn=utils.seed_worker,
                                    generator=g, shuffle=True)
    
    # Initializing the model
    llama = Llama(relations, args.model_id, args.repo_token, args.attention_dropout)
    model = llama.get_model()
    model.resize_token_embeddings(training_set.tokenizer_len())
    model.to(device)
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate )
    scheduler = lr_scheduler.StepLR(optimizer, gamma=args.decay, step_size = 1)

    v_loss = 1_000_000
    no_change_counter = 1
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}\n-------------------------------')
        lr = optimizer.param_groups[0]['lr']
        model.train()
        loop = tqdm(training_generator, disable = not args.verbose)

        # Loop over batches within epoch using DataLoader
        for _, data in enumerate(loop):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = loss_f(logits , data[1].to(device))
            loss.backward()       
            optimizer.step()
            last_loss = loss.item()
        
        v_losses = []
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(validation_generator):
                inputs = data[0].to(device)
                logits = model(**inputs).logits
                loss = loss_f(logits , data[1].to(device))
                v_losses.append(loss)
            
            v_loss_epoch = sum(v_losses) / len(v_losses)
            utils.write_log(f'lr {lr:8f} train loss {last_loss:.8f} val loss {v_loss_epoch:.8f}')

            if v_loss - v_loss_epoch > 0.00001:
                v_loss = v_loss_epoch
                no_change_counter = 0
                torch.save(model.state_dict(), utils.get_timestamp()+'chkpnt.pt')
            elif no_change_counter > args.patience - 1:
                break
            else:
                no_change_counter += 1
        
        scheduler.step()

if __name__ == "__main__": main()
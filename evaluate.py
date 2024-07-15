from datetime import datetime
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils import Utils
from model import KGDataset,Llama
from torch.utils.data import DataLoader

def main():
    """
    """

    def load_evaluation_triples(triples, data_set):
        """
        """
        
        for i, _ in enumerate(data_set):
            item = data_set.gettext(i)
            key = item[3] + '_' + item[4]
            rel_list = triples.setdefault(key, [])
            rel_index = data_set.relations.index(item[1])
            rel_list.append(rel_index)
            triples[key] = rel_list
        
        return triples

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

    # Initialize metrices
    ranks = []
    ranks_filtered = []
    mrr = []
    mrr_filtered = []
    hits = []
    hits_filtered = []
    for i in range(10):
            hits.append([])
            hits_filtered.append([])
    
    # Load datasets
    all_triples = {}
    training_set = KGDataset(args=args,
                             relations=relations,
                             triples_filename='train.tsv')
    all_triples = load_evaluation_triples(all_triples, training_set)
    del training_set
    
    validation_set = KGDataset(args=args,
                               relations=relations,
                               triples_filename='dev.tsv')
    all_triples = load_evaluation_triples(all_triples, validation_set)
    del validation_set
    
    testing_set = KGDataset(args=args,
                               relations=relations,
                               triples_filename='test.tsv')
    all_triples = load_evaluation_triples(all_triples, testing_set)
    testing_generator = DataLoader(testing_set,
                                      batch_size = args.batch_size,
                                    worker_init_fn=utils.seed_worker,
                                    generator=g,)
    
    # Initialize Llama
    llama = Llama(relations, args.model_id, args.repo_token, 0.0)
    model = llama.get_model()
    model.resize_token_embeddings(testing_set.tokenizer_len())
    if args.checkpoint_path == '':
        model.load_state_dict(torch.load(utils.get_timestamp()+'chkpnt.pt', ))
    else:
        model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)

    model.eval()
    with torch.no_grad():
        loop = tqdm(testing_generator, disable = not args.verbose)
        for id, data in enumerate(loop):
            inputs = data[0].to(device)
            data[1] = data[1].to(device)
            logits = model(**inputs).logits
            for i, item in enumerate(logits):
                gold_index = torch.argmax(data[1][i])
                indices = torch.argsort(item, descending = True)
                rank = (indices==gold_index).nonzero().item()
                item_id = id * args.batch_size + i

                # description evaluation
                if rank > 3:
                    data_0, data_1 = testing_set.getitem_w_description(item_id)
                    data_0['input_ids'] = torch.unsqueeze(data_0['input_ids'], 0)
                    data_0['attention_mask'] = torch.unsqueeze(data_0['attention_mask'], 0)
                    data_0 = data_0.to(device)
                    # data_1 = data_1.to(device)
                    # gold_index2 = torch.argmax(data_1)
                    logits2 = model(**data_0).logits
                    indices2 = torch.argsort(logits2[0], descending = True)
                    new_rank = (indices2==gold_index).nonzero().item()
                    if new_rank < rank:
                        rank = new_rank
                        indices = indices2
                
                ranks.append(rank + 1)
                mrr.append(1/(rank + 1))

                # filter work
                filter_rank = rank

                # get higher predicted relations
                # indices_list = indices.view(-1)
                indices_list = indices.tolist()
                higher_rels = indices_list[: indices_list.index(gold_index) ]
                
                # get gold relations for the triple
                triple = testing_set.gettext(item_id)
                key = triple[3] + '_' + triple[4]
                rel_list = all_triples[key]

                # loop higher rels
                for j, rel_id in enumerate(higher_rels):
                    if rel_id in rel_list:
                        filter_rank -= 1
                
                ranks_filtered.append(filter_rank + 1)
                mrr_filtered.append(1/(filter_rank + 1))

                # Hits work
                failure_str = ''
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                    
                    if filter_rank <= hits_level:
                        hits_filtered[hits_level].append(1.0)
                    else:
                        hits_filtered[hits_level].append(0.0)
                    
                    if rank > 10 and hits_level == 9:
                        h, r, t, e1, e2 = testing_set.gettext(item_id)
                        failure_str += str(rank) + '^' +str(item_id) + '^' + h + '^' + r + '^' + t + '^' + e1 + '^' + e2
                
                utils.write_log(failure_str,'failures.out', False)

    utils.write_log(f'\n{"MR":<15}: {np.mean(ranks):.4f}')
    utils.write_log(f'{"MR Filtered":<15}: {np.mean(ranks_filtered):.4f}')
    utils.write_log(f'\n{"MRR":<15}: {np.mean(mrr):.4f}')
    utils.write_log(f'{"MRR Filtered":<15}: {np.mean(mrr_filtered):.4f}')
    for i in [0,4,9]:
        utils.write_log(f'Raw Hits           {i + 1:<3}: {np.mean(hits[i]):<5.6f}')
        utils.write_log(f'Raw Filtered Hits  {i + 1:<3}: {np.mean(hits_filtered[i]):<5.6f}')

if __name__ == "__main__": main()

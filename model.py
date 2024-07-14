class Llama():
    """
    """

    def __init__(self,relations, model_id, repo_token, attention_dropout) -> None:
        """
        """

        self.relations = relations
        self.model_id = model_id
        self.repo_token = repo_token
        self.attention_dropout = attention_dropout
        
    
    def get_model(self):
        configuration = LlamaConfig(attention_dropout = self.attention_dropout)
        configuration.num_labels = len(self.relations)
        model = LlamaForSequenceClassification.from_pretrained(
            self.model_id,
            config = configuration,
            token = self.repo_token,
            
            )
        model.config.pad_token_id = 4
        for param in model.parameters():
            if param.dtype == torch.float32 or \
            param.dtype == torch.float16 :
                param.data = param.data.to(torch.bfloat16)

        return model

class KGDataset(Dataset):
    """
    """

    def __init__(self, args,
                 triples_filename: str,
                 relations: list,
                 is_training= False,
                ) -> None:
        """This constructor loads the necessary data.
        """

        self.args = args
        self.relations = relations
        self.cache = {}

        # Read file
        f = open( os.path.join(args.dataset_directory, triples_filename) )
        file_lines = f.readlines()
        f.close()

        # Load lines based on the specified data amount
        if is_training:
            self.lines = file_lines[: int(len(file_lines) * args.data_size) ]
        else:
            self.lines = file_lines
        print(f'\n{len(self.lines)} {triples_filename} triples loaded')

        # Tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_id,
                                                   token = self.args.repo_token,)
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.model_max_length = args.padding
        self.tokenizer = tokenizer

        # Load entity translations
        self.entities_dict = {}
        with open( os.path.join(args.dataset_directory, args.entities_filename) ) as f:
            for line in f.readlines():
                fields = line.split('\t')
                fields = [p.strip() for p in fields]
                self.entities_dict[ fields[0] ] = fields[1]
        
        # Load entity translations
        self.descriptions_dict = {}
        with open( os.path.join(args.dataset_directory, args.descriptions_filename) ) as f:
            for line in f.readlines():
                fields = line.split('\t')
                fields = [p.strip() for p in fields]
                self.descriptions_dict[ fields[0] ] = fields[1]
        
        print(f'\n{len(self.entities_dict.keys())} entity translations loaded')

        return None
    
    def __len__(self) -> int:
        """
        """

        return len(self.lines)
    
    def gettext(self, index):
        """
        """

        fields = self.lines[index].split('\t')
        fields = [p.strip() for p in fields]
        head = self.entities_dict[fields[0]]
        tail = self.entities_dict[fields[2]]
        rel = fields[1]
        return head, rel, tail, fields[0],fields[2]
    
    def tokenizer_len(self):
        """
        """

        return len(self.tokenizer)
    
    def __getitem__(self, index):
        """
        """

        # Check the cache
        if index in self.cache.keys():
            return self.cache[index]

        # Create triple from a dataset line

        fields = self.lines[index].split('\t')
        fields = [p.strip() for p in fields]

        # Prepare Y label
        rel = fields[1]
        rel_index = self.relations.index(rel)
        relations_tagged = [0.0] * len(self.relations)
        relations_tagged[ rel_index ] = 1.0

        # Tokenize
        inputs = self.tokenizer([[self.entities_dict[fields[0]],\
                                   self.entities_dict[fields[2]]]],
                  padding='max_length',
                  truncation = True,
                  return_attention_mask=True,
                  return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].squeeze(0).squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0).squeeze(0)

        result = (inputs, torch.tensor(relations_tagged))

        self.cache[index] = result
        
        return result
    
    def getitem_w_description(self, index):
        """
        """
        
        fields = self.lines[index].split('\t')
        fields = [p.strip() for p in fields]

        # Prepare Y label
        rel = fields[1]
        rel_index = self.relations.index(rel)
        relations_tagged = [0.0] * len(self.relations)
        relations_tagged[ rel_index ] = 1.0

        # Tokenize
        inputs = self.tokenizer([[self.descriptions_dict[fields[0]],\
                                   self.descriptions_dict[fields[2]]]],
                  padding='max_length',
                  truncation = True,
                  return_attention_mask=True,
                  return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].squeeze(0).squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0).squeeze(0)

        result = (inputs, torch.tensor(relations_tagged))
        
        return result
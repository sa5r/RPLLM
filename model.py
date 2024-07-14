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
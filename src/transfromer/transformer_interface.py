from abc import ABC, abstractmethod

class AbstractTransformer(ABC):
    def __init__(self):
        """"""
        pass
    def token_embedding(self,):
        """
        Return a Seqeunce of Vectors from the Input Token ID Sequence
        
        Maps: (batch_size, sequence_length) -> (batch_size, sequence_length, d_model)
        """
        pass 

    def transformer_block(self,):
        """
        A transformer block: 

        Maps: (batch_size, sequence_length, d_model) -> (batch_size, sequence_length,d_model)

        Where to paralle?
            1. Batch-wise: Same Transformer forward operation on each batch of the data.
            2. Element-wise: Same RMSNorm && FFN applied on each position of a sequence.
            3. Att_Head-wise: Same attention operations across heads in a multi-head attention operation.
        """
        pass

    def normalization_block(self,):
        """
        A normaliztion block:

        Maps: 
        """
        pass

    def pred_next(self,):
        """
        Return next-token logits for prediction, using a standard learned linear 
        transformation to convert the output of the Transformer blocks.
        """
        pass

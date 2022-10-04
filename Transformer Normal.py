class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads): # change name embed_size,heads - we have an embedding and we are going to split it in (heads) =! parts
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads."
        
        #1. in __init__:  self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False) (same for key and value weights)
        #2. in forward: put "queries = self.queries(queries)" above "queries = queries.reshape(...)" (also same for keys and values)
        # before - self.head_dim on self.embed_size
        
        self.values = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.fc_out = nn.Linear(head * self.head_dim, embed_size) # concatenate
        
    def forward(self, values, keys, queries, mask): # DUV mask
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # do linear
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # split embedding into each heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) 
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim) 
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        
        # calculate QK^T
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask = 0, value = float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        
        out = self.fc_out(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Position-wise Feed-Forward Network (2)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        # applying dropout after the feed forward network and the mha to prevent overfitting
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query)) # + -> skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device # DUV - what is device?
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size) # DUV
        self.position_embedding = nn.Embedding(max_length, embed_size) # DUV
        
        self.layers = nn.ModuleList( # DUV - modulelist
            [
                TransformersBlock(embed_size, heads, dropout = dropout, forward_expansion = forward_expansion)
            for _ in range(num_layers)]
        )
        self.dropout = nn.Droupout(droupout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        # getting positions
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # input embedding + positional encoding
        
        for layer in self.layers:
            out = layer(out, out, out, mask) # key, value, query are all the same = out
            
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Droupout(dropout)
        
    def forward(self, x, value, key, source_mask, target_mask): # source mask and target mask difference - 1st is the one for the encoder (opt.), 2nd is for the decoder
        attention = self.attention(x, x, x, target_mask) # all the 3 inputs - k, q, v are the same
        query = self.droupout(self.norm(attention + x)) # skip connection; what we obtain from this block is the query for the next
        out = self.transformer_block(value, key, query, source_mask)
        
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_si, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(mebed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )
        
        self.fc_out = nn.Linear(embed_size, target_vocab_size) # concatenate
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, source_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, source_mask, target_mask) # k and v are the output of the encoder, q is x
            
        out = self.fc_out(x)
        
        return out


class Transformer(nn.Module):
    def __init__(self, source_vocabe_size, target_vocab_size, source_pad_index, target_pad_index, embed_size = 256, num_layers = 6, forward_expansion = 4, heads = 8, dropout = 0, device = "cuda", max_length = 100):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        
        self.source_pad_index = source_pad_index 
        self.target_pag_index = target_pad_index # DUV what is pad?
        self.device = device
        
    def make_source_mask(self, source):
        # if is a source_pad_index is going to be set to 0, and if not 1 
        source_mask = (source != self.source_pad_index).unsqueeze(1).unsqueeze(2) # DUV
        # source mask shape: (N, 1, 1, source_len)
        
        return source_mask.to(self.device)
    
    def make_target_mask(self, target):
        N, target_len = target.shape
        
        # creating a triangular matrix with 1 and expand
        target_mask = torch.tril(torch.ones(target_len, target_len)).expand(N, 1, target_len, target_len)
        
        return target_mask.to(self.device)
    
    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.target_source_mask(target)
        enc_source = self.encoder(self, source_mask)
        out = self.decoder(target, enc_source, source_mask, target_mask)
        
        return out

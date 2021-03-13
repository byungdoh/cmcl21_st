from transformers import RobertaModel
import torch
import torch.nn as nn

class RobertaForGazePrediction(nn.Module):
    """ Custom model for RoBERTa-based gaze duration prediction"""

    def __init__(
        self,
        pretrained,
        input_dim,
        dropout_1,
        hidden_dim,
        activation,
        dropout_2,
    ):
        super(RobertaForGazePrediction, self).__init__()
        self.pretrained_model = pretrained
        self.dropout_1 = nn.Dropout(dropout_1)
        self.dense = nn.Linear(input_dim+2, hidden_dim)
        self.activation = activation
        self.dropout_2 = nn.Dropout(dropout_2)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.activation_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_idx=None,
        wlen=None,
        prop=None,
        ablate_wlen=False,
        ablate_prop=False
    ):

        # outputs: a tuple of torch.FloatTensor comprising various elements
        # depending on the configuration (RobertaConfig) and inputs
        outputs = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        # Sequence of hidden-states at the output of the last layer of the model
        x = outputs[0]
        x = torch.cat([x[i][first_idx[i]] for i in range(len(x))], dim=0)

        if not ablate_wlen and not ablate_prop:
            x = torch.cat([x,wlen,prop], dim=1)
        elif not ablate_wlen and ablate_prop:
            x = torch.cat([x,wlen], dim=1)
        elif ablate_wlen and not ablate_prop:
            x = torch.cat([x,prop], dim=1)

        x = self.dropout_1(x)
        x = self.dense(x)
        x = self.activation_fn[self.activation](x)
        x = self.dropout_2(x)
        x = self.out_proj(x)

        return x
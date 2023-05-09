import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss


class GRU4RecExt(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GRU4RecExt, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        #self.selected_features = config['selected_features']
        #self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        #self.num_feature_field = len(config['selected_features'])

        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        pretrained_news_emb = dataset.get_preload_weight('nid')
        self.feature_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_news_emb).to(dtype=torch.float))
        self.feature_embed_size = self.feature_embed_layer.embedding_dim

        self.item_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # For simplicity, we use same architecture for item_gru and feature_gru
        self.feature_gru_layers = nn.GRU(
            input_size=self.feature_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense_layer = nn.Linear(self.hidden_size * 2, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['feature_embed_layer']

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.dropout(item_seq_emb)
        item_gru_output, _ = self.item_gru_layers(item_seq_emb_dropout)  # [B Len H]

        feature_seq_emb = self.feature_embed_layer(item_seq)
        feature_gru_output, _ = self.feature_gru_layers(feature_seq_emb)  # [B Len H]

        output_concat = torch.cat((item_gru_output, feature_gru_output), -1)  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    

import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import load_config

import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

def run_recbole(config_dict):
    config = Config(model='GRU4Rec', dataset='mind_small', config_dict=config_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # write config info into log
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = GRU4RecExt(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, 
                                                      valid_data=valid_data, 
                                                      show_progress=True)
    return best_valid_score, best_valid_result



if __name__ == '__main__':
    config_dict = load_config('configs/sequential/gru4rec_ext.yaml')
    run_recbole(config_dict=config_dict)
    
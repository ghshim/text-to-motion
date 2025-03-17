from options.base_option import BaseOptions

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        
        # Data options
        self.parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
        self.parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
        
        # General options
        self.parser.add_argument("--seed", default=1000, type=int, help="Seed")
        self.parser.add_argument('--is_continue', action='store_true', help='Continue training from checkpoint')
        self.parser.add_argument('--log_every', type=int, default=50, help='Log frequency (in iterations)')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Evaluation frequency (in epochs)')
        self.parser.add_argument('--save_latest', type=int, default=300, help='Frequency for saving the latest checkpoint (in iterations)')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--resume_path', type=str, default=None, help='Checkpoint path')
        
        # Training options
        self.parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
        self.parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of text")

        # Model options
        self.parser.add_argument("--num_heads", type=int, default=8, help="Num heads of transformer")
        self.parser.add_argument("--dim_model", type=int, default=512, help="Dimension of motion")
        self.parser.add_argument("--num_tokens", type=int, default=263, help="Dimension of motion")
        self.parser.add_argument("--dim_text", type=int, default=512, help="Dimension of clip text embedding")
        self.parser.add_argument("--dim_motion", type=int, default=263, help="Dimension of motion")
        self.parser.add_argument("--num_encoder_layers", type=int, default=8, help="Dimension of motion")
        self.parser.add_argument("--num_decoder_layers", type=int, default=8, help="Dimension of motion")
        self.parser.add_argument("--dropout_p", type=float, default=0.1, help="Dimension of motion")
        
        self.is_train = True
    
from modules.hoi4abot.hoibot.modules.transformer.dual_crosstransformer import DualTransformer
from modules.hoi4abot.hoibot.modules.transformer.SC_transformer import SC_transformer
from modules.hoi4abot.hoibot.modules.transformer.Temp_transformer import TempTransformer

def get_transformer(transformer_type, embedding_dimension, sliding_window,
                    depth,semantic_type, mlp_ratio, drop_rate, num_heads,
                    pos_embed_type, image_cls_type, use_hoi_token, mainbranch, is_entity=False, no_reduction=False, just_the_passed=False):
    if transformer_type in ["sc"]:
        if just_the_passed:
            embed_dim =embedding_dimension
        else:
            embed_dim = 3 * embedding_dimension if no_reduction else 2*embedding_dimension
        transformer = SC_transformer(embed_dim=embed_dim,
                                          windows_size=sliding_window,
                                          depth=depth,  # number of blocks
                                          dual_transformer_type=transformer_type,
                                          semantic_type=semantic_type,
                                          mlp_ratio=mlp_ratio,
                                          drop_rate=drop_rate,
                                          num_heads=num_heads,
                                          learnable_pos_embed=pos_embed_type,
                                          image_cls_type=image_cls_type,
                                          use_hoi_token=use_hoi_token,
                                          isquery=mainbranch,
                                          is_entity=is_entity,
                                          )
    elif transformer_type in ["stacked"]:
        embed_dim = embedding_dimension  if just_the_passed else 2*embedding_dimension
        transformer = TempTransformer(embed_dim=embed_dim,
                        windows_size=sliding_window,
                        depth=depth,  # number of blocks
                        dual_transformer_type=transformer_type,
                        semantic_type=semantic_type,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        num_heads=num_heads,
                        learnable_pos_embed=pos_embed_type,
                        image_cls_type=image_cls_type,
                        use_hoi_token=use_hoi_token,
                        isquery=mainbranch,
                        is_entity=False,
                        )
    else:
        embed_dim = embedding_dimension  if just_the_passed else 2*embedding_dimension
        transformer = DualTransformer(embed_dim=embed_dim,
                                           windows_size=sliding_window,
                                           depth=depth,  # number of blocks
                                           dual_transformer_type=transformer_type,
                                           semantic_type=semantic_type,
                                           mlp_ratio=mlp_ratio,
                                           drop_rate=drop_rate,
                                           num_heads=num_heads,
                                           learnable_pos_embed=pos_embed_type,
                                           image_cls_type=image_cls_type,
                                           use_hoi_token=use_hoi_token,
                                           mainbranch=mainbranch
                                           )
    return transformer
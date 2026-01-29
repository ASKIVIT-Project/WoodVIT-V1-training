# -*- coding: utf-8 -*-
__author__ = "Felix Niemöller"
__email__ = "usnrb@student.kit.edu"

""" 
Maps labels to integers. 

Usage example:
from torchaskivit.constants import ASKIVIT_LABELS_V1
"""

## ****************************************************************
## * ASKIVIT V1.4                                                 *
## ****************************************************************
ASKIVIT_LABELS_V1 = {
    "Holz_Massivholz": 0,
    "Holz_Sperrholz": 1,
    "Holz_Spanplatte": 2,
    "Holz_Mitteldichte-Faserplatte": 3,

    "Nicht-Holz_Hintergrund": 4,
    "Nicht-Holz_Metall": 5,
    "Nicht-Holz_Kunststoff": 6,
    "Nicht-Holz_Mineralik": 7,
    "Nicht-Holz_Polster": 8,
    "Nicht-Holz_Holz-verdeckt-durch-Polster": 9,
    "Nicht-Holz_Holz-verdeckt-durch-Karton": 10,
    "Nicht-Holz_Holz-verdeckt-durch-Kunststoff": 11,
    "Nicht-Holz_Holz-verdeckt-durch-Mineralik": 12,
    "Nicht-Holz_Metall-verdeckt-durch-Holz": 13,
    "Nicht-Holz_Metall-verdeckt-durch-Polster": 14,
    "Nicht-Holz_Metall-verdeckt-durch-Kunststoff": 15,
}

ASKIVIT_LABELS_V1_MAIN = {
    "Holz": 0,
    "Nicht-Holz": 1,
}

ASKIVIT_LABELS_V1_ENG = [
    "Solid Wood",
    "Plywood",
    "Chipboard",
    "MDF",

    "Background",
    "Metal",
    "Plastic",
    "Mineralics",
    "Upholstery",
    "Wood covered by Upholstery",
    "Wood covered by Cardboard",
    "Wood covered by Plastic",
    "Wood covered by Mineralics",
    "Metal covered by Wood",
    "Metal covered by Upholstery",
    "Metal covered by Plastic",

]

ASKIVIT_LABELS_V1_ABR_ENG = [
    "Solid Wood",
    "Plywood",
    "Chipboard",
    "MDF",

    "Background",
    "Metal",
    "Plastic",
    "Mineral",
    "Upholstery",
    "WcbUpholstery",
    "WcbCardboard",
    "WcbPlastic",
    "WcbMineral",
    "McbWood",
    "McbUpholstery",
    "McbPlastic",
]


## ****************************************************************
## * ASKIVIT V2                                                   *
## ****************************************************************

ASKIVIT_LABELS_V2 = {
    "Background": 0,
    "Chipboard": 1,
    "Solid_wood": 2,
    "Medium_density_fiberboard": 3,
    "Oriented_strand_board": 4,
    "Other_Wood": 5,
    "Covered_solid_wood_under_plastic": 6,
    "Metal": 7,
    "Plastic": 8,
    "Ceramics": 9,
    "Other_non_wood": 10,
    "Covered_metal_under_chipboard": 11,
    "Covered_metal_under_solid_wood": 12,
    "Covered_metal_under_plastic": 13
}

ASKIVIT_LABELS_V2_MAIN = {
    "Wood": 0,
    "Non_wood": 1,
}


## ****************************************************************
## * ASKIVIT V2 metal expert                                                   *
## ****************************************************************

ASKIVIT_LABELS_V2_METAL_EXPERT = {
    "Background": 0,
    "Chipboard": 1,
    "Solid_wood": 2,
    "Medium_density_fiberboard": 3,
    "Oriented_strand_board": 4,
    "Other_Wood": 5,
    "Covered_solid_wood_under_plastic": 6,
    "Metal": 7,
    "Plastic": 8,
    "Ceramics": 9,
    "Other_non_wood": 10,
    "Covered_metal_under_chipboard": 11,
    "Covered_metal_under_solid_wood": 12,
    "Covered_metal_under_plastic": 13
}

ASKIVIT_LABELS_V2_MAIN_METAL_EXPERT = {
    "Metal": 0,
    "Non_metal": 1,
}


## ****************************************************************
## * ASKIVIT V1.5                                                  *
## ****************************************************************

# labels for distinction of files (including underscores)
ASKIVIT_LABELS_V1_5 = {
    "Solid_Wood": 0,
    "Plywood": 1,
    "Chipboard": 2,
    "MDF": 3,

    "Background": 4,
    "Metal": 5,
    "Plastic": 6,
    "Mineralics": 7,
    "Upholstery": 8,
    "Wood_covered_by_Upholstery": 9,
    "Wood_covered_by_Cardboard": 10,
    "Wood_covered_by_Plastic": 11,
    "Wood_covered_by_Mineralics": 12,
    "Metal_covered_by_Wood": 13,
    "Metal_covered_by_Upholstery": 14,
    "Metal_covered_by_Plastic": 15,
}

ASKIVIT_LABELS_V1_5_MAIN = {
    "Wood": 0,
    "Non_wood": 1,
}

# labels for plotting confusion matrix (without underscores)
ASKIVIT_LABELS_V1_5_PLOT = {
    "Solid Wood": 0,
    "Plywood": 1,
    "Chipboard": 2,
    "MDF": 3,

    "Background": 4,
    "Metal": 5,
    "Plastic": 6,
    "Minerals": 7,
    "Upholstery": 8,
    "Wood cov. by Upholstery": 9,
    "Wood cov. by Cardboard": 10,
    "Wood cov. by Plastic": 11,
    "Wood cov. by Minerals": 12,
    "Metal cov. by Wood": 13,
    "Metal cov. by Upholstery": 14,
    "Metal cov. by Plastic": 15,
}

ASKIVIT_LABELS_V1_5_MAIN_PLOT = {
    "Wood": 0,
    "Non wood": 1,
}


## ****************************************************************
## * ASKIVIT V1.5 metal expert                                                  *
## ****************************************************************

ASKIVIT_LABELS_V1_5_METAL_EXPERT = {
    "Solid_Wood": 0,
    "Plywood": 1,
    "Chipboard": 2,
    "MDF": 3,

    "Background": 4,
    "Metal": 5,
    "Plastic": 6,
    "Mineralics": 7,
    "Upholstery": 8,
    "Wood_covered_by_Upholstery": 9,
    "Wood_covered_by_Cardboard": 10,
    "Wood_covered_by_Plastic": 11,
    "Wood_covered_by_Mineralics": 12,
    "Metal_covered_by_Wood": 13,
    "Metal_covered_by_Upholstery": 14,
    "Metal_covered_by_Plastic": 15,
}

ASKIVIT_LABELS_V1_5_METAL_EXPERT_MAIN = {
    "Metal": 0,
    "Non_metal": 1,
}
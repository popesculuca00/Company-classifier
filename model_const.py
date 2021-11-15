import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu( self.conv(x) )

class Classifier(nn.Module):
    def __init__(self, alphabet_size, max_length, dropout, n_classes=148):
        super(Classifier, self).__init__()
        self.do = nn.Dropout2d(dropout)
        self.conv1 = ConvBlock(in_channels=alphabet_size,
                               out_channels=1024,
                               kernel_size=7)
        self.mp1 = nn.MaxPool1d(3)
        
        self.conv2 = ConvBlock(in_channels=1024,
                               out_channels=1024,
                               kernel_size=7)
        self.mp2 = nn.MaxPool1d(3)

        self.conv3 = ConvBlock(in_channels=1024,
                               out_channels=1024,
                               kernel_size = 3)

        self.conv4 = ConvBlock(in_channels=1024,
                               out_channels=1024,
                               kernel_size = 3)

        self.conv5 = ConvBlock(in_channels=1024,
                               out_channels=1024,
                               kernel_size = 3)

        self.conv6 = ConvBlock(in_channels=1024,
                               out_channels=1024,
                               kernel_size = 3)
        self.mp3 = nn.MaxPool1d(3)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential( nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5) )
        self.fc2 = nn.Sequential( nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5) )
        self.fc3 = nn.Sequential( nn.Linear(1024, n_classes))
        
    def forward(self, x):
        x = self.do(x)
        x = self.mp1( self.conv1(x) )
        x = self.mp2( self.conv2(x) )
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mp3( self.conv6(x) )
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


industry_decoder = {0: 'accounting',
                    1: 'e-learning',
                    2: 'machinery',
                    3: 'computer software',
                    4: 'construction',
                    5: 'consumer electronics',
                    6: 'consumer goods',
                    7: 'hospitality',
                    8: 'consumer services',
                    9: 'automotive',
                    10: 'cosmetics',
                    11: 'arts and crafts',
                    12: 'education management',
                    13: 'aviation & aerospace',
                    14: 'architecture & planning',
                    15: 'electrical/electronic manufacturing',
                    16: 'environmental services',
                    17: 'human resources',
                    18: 'writing and editing',
                    19: 'individual & family services',
                    20: 'pharmaceuticals',
                    21: 'photography',
                    22: 'plastics',
                    23: 'primary/secondary education',
                    24: 'computer & network security',
                    25: 'banking',
                    26: 'professional training & coaching',
                    27: 'health, wellness and fitness',
                    28: 'financial services',
                    29: 'farming',
                    30: 'facilities services',
                    31: 'broadcast media',
                    32: 'apparel & fashion',
                    33: 'food & beverages',
                    34: 'food production',
                    35: 'furniture',
                    36: 'government administration',
                    37: 'graphic design',
                    38: 'higher education',
                    39: 'civil engineering',
                    40: 'hospital & health care',
                    41: 'fine art',
                    42: 'events services',
                    43: 'design',
                    44: 'entertainment',
                    45: 'building materials',
                    46: 'business supplies and equipment',
                    47: 'biotechnology',
                    48: 'chemicals',
                    49: 'civic & social organization',
                    50: 'printing',
                    51: 'computer games',
                    52: 'public relations and communications',
                    53: 'management consulting',
                    54: 'international trade and development',
                    55: 'internet',
                    56: 'investment management',
                    57: 'law practice',
                    58: 'legal services',
                    59: 'airlines/aviation',
                    60: 'leisure, travel & tourism',
                    61: 'logistics and supply chain',
                    62: 'import and export',
                    63: 'luxury goods & jewelry',
                    64: 'maritime',
                    65: 'information technology and services',
                    66: 'marketing and advertising',
                    67: 'mechanical or industrial engineering',
                    68: 'media production',
                    69: 'medical devices',
                    70: 'medical practice',
                    71: 'mental health care',
                    72: 'mining & metals',
                    73: 'motion pictures and film',
                    74: 'music',
                    75: 'publishing',
                    76: 'insurance',
                    77: 'non-profit organization management',
                    78: 'information services',
                    79: 'telecommunications',
                    80: 'real estate',
                    81: 'religious institutions',
                    82: 'renewables & environment',
                    83: 'research',
                    84: 'restaurants',
                    85: 'industrial automation',
                    86: 'security and investigations',
                    87: 'sporting goods',
                    88: 'sports',
                    89: 'staffing and recruiting',
                    90: 'retail',
                    91: 'textiles',
                    92: 'transportation/trucking/railroad',
                    93: 'utilities',
                    94: 'venture capital & private equity',
                    95: 'wholesale',
                    96: 'wine and spirits',
                    97: 'performing arts',
                    98: 'packaging and containers',
                    99: 'online media',
                    100: 'oil & energy',
                    101: 'market research',
                    102: 'commercial real estate',
                    103: 'veterinary',
                    104: 'computer hardware',
                    105: 'computer networking',
                    106: 'recreational facilities and services',
                    107: 'outsourcing/offshoring',
                    108: 'executive office',
                    109: 'program development',
                    110: 'translation and localization',
                    111: 'philanthropy',
                    112: 'newspapers',
                    113: 'public safety',
                    114: 'defense & space',
                    115: 'museums and institutions',
                    116: 'investment banking',
                    117: 'government relations',
                    118: 'law enforcement',
                    119: 'fund-raising',
                    120: 'think tanks',
                    121: 'alternative medicine',
                    122: 'warehousing',
                    123: 'international affairs',
                    124: 'semiconductors',
                    125: 'animation',
                    126: 'public policy',
                    127: 'political organization',
                    128: 'paper & forest products',
                    129: 'wireless',
                    130: 'nonprofit organization management',
                    131: 'capital markets',
                    132: 'glass, ceramics & concrete',
                    133: 'libraries',
                    134: 'gambling & casinos',
                    135: 'judiciary',
                    136: 'military',
                    137: 'shipbuilding',
                    138: 'package/freight delivery',
                    139: 'dairy',
                    140: 'supermarkets',
                    141: 'alternative dispute resolution',
                    142: 'nanotechnology',
                    143: 'fishery',
                    144: 'ranching',
                    145: 'railroad manufacture',
                    146: 'tobacco',
                    147: 'legislative office'}
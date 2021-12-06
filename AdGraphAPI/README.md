Parse timeline files: `python2 rules_parser.py`

Extract feature values: `./adgraph /home/shitong/rendering_stream/ data/ features/ mappings/ www.google.com/`

Example directory tree:
```
├── data
│   └── www.google.com
│       ├── log_www.google.com_1565674017.483595.json
│       └── parsed_log_www.google.com_1565674017.483595.json
├── features
│   └── www.google.com.csv
├── filterlists
│   ├── easylist.txt
│   ├── easyprivacy.txt
│   ├── fanboyannoyance.txt
│   ├── killer.txt
│   └── warning.txt
└── mappings
    ├── www.google.com.csv
    └── www.google.com.json
```

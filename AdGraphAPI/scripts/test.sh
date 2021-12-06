# $2 is the prefix of the path, if not specified then we use the default
# path in AdGraphAPI code
if [ "$2" = "parse-unmod" ]; then
	python3 ~/AdGraphAPI/scripts/load_page_adgraph.py --domain $1 --id $3 --final-domain $4 --mode proxy --strategy $5
	python ~/AdGraphAPI/scripts/rules_parser.py --target-dir ~/rendering_stream/timeline --domain $1 --strategy $5
	~/AdGraphAPI/adgraph ~/rendering_stream/ features/ mappings/ $1 parsed_$5_$1
else
	python3 ~/AdGraphAPI/scripts/load_page_adgraph.py --domain $1 --load-modified --id $3 --final-domain $4 --mode proxy --strategy $5
	python ~/AdGraphAPI/scripts/rules_parser.py --target-dir ~/rendering_stream/timeline --domain $1 --parse-modified --strategy $5
	~/AdGraphAPI/adgraph ~/rendering_stream/ features/ mappings/ $1 parsed_modified_$5_$1
fi

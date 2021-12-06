#include <iostream>
#include "utilities.h"
#include "json_to_adgraph_parser.h"

int main(int argc, char *argv[])
{
    std::string base_directory = argv[1];
    std::string features_directory = base_directory + argv[2];
    std::string mapping_directory = base_directory + argv[3];
    std::string domain = argv[4];
    std::string timeline_filename = argv[5];
    std::string timeline_filepath = base_directory + "timeline/" + timeline_filename + ".json";

    std::cout << "Base: " << base_directory << std::endl;
    std::cout << "Features: " << features_directory << std::endl;
    std::cout << "Mapping: " << mapping_directory << std::endl;
    std::cout << "Domain: " << domain << std::endl;
    std::cout << "Timeline file: " << timeline_filepath << std::endl;

    AdGraphAPI::Utilities::ordered_json json_content;
    json_content = AdGraphAPI::Utilities::ReadJSON(timeline_filepath);

    AdGraphAPI::JSONToAdGraphParser parser_object(json_content["url"].get<std::string>(),
		                                  features_directory + domain + ".csv",
						  mapping_directory + domain + ".csv");
    parser_object.CreateGraph(json_content);
}

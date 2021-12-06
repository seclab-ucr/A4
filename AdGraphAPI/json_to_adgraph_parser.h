#ifndef JSONTOADGRAPHPARSER
#define JSONTOADGRAPHPARSER
#include <string>
#include <vector>
#include <map>

#include "utilities.h"
#include "adgraph.h"
#include "events_and_properties.h"

namespace AdGraphAPI
{

  class JSONToAdGraphParser
  {
  public:
    JSONToAdGraphParser(std::string base_domain, std::string features_file_name, std::string url_id_string_map_file_name);
    void CreateGraph(Utilities::ordered_json json_content);

  protected:
    AdGraph adgraph_;
    std::string features_file_name_;
    std::string url_id_string_map_file_name_;
    std::string visualization_file_name_;
    std::string timing_file_name_;
    std::tuple<std::string, bool, bool, bool, std::vector<std::string>, std::string> ExtractJSONPropertiesForHTMLNode(Utilities::ordered_json json_item);
    std::tuple<bool, std::string, bool, std::string> ExtractJSONPropertiesForHTTPNode(Utilities::ordered_json json_item);
    std::tuple<std::string, std::string> ExtractJSONPropertiesAttributes(Utilities::ordered_json json_item);
  };

} // namespace AdGraphAPI
#endif // JSONTOADGRAPHPARSER
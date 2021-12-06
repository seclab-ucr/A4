#include "json_to_adgraph_parser.h"
#include <tuple>

namespace AdGraphAPI
{
    JSONToAdGraphParser::JSONToAdGraphParser(std::string base_domain,
                                             std::string features_file_name,
                                             std::string url_id_string_map_file_name) : adgraph_(base_domain),
                                                                                        features_file_name_(features_file_name),
                                                                                        url_id_string_map_file_name_(url_id_string_map_file_name) {}

    std::tuple<std::string, bool, bool, bool, std::vector<std::string>, std::string> JSONToAdGraphParser::ExtractJSONPropertiesForHTMLNode(Utilities::ordered_json json_item)
    {
        std::string tag_name = "";
        bool script_is_active = (json_item[ACTOR_ID].get<std::string>() == "0") ? false : true;
        bool async = false;
        bool defer = false;
        std::vector<std::string> attribute_name_and_values;
        std::string previous_sibling_id = "0";

        if (json_item[EVENT_TYPE] == NODE_INSERTION)
        {
            previous_sibling_id = json_item["node_previous_sibling_id"].get<std::string>();
            for (auto &json_attr : json_item["node_attributes"])
            {
                attribute_name_and_values.push_back(json_attr["attr_name"].get<std::string>());
                attribute_name_and_values.push_back(json_attr["attr_value"].get<std::string>());
            }
        }

        if (json_item.find(TAG_NAME) != json_item.end())
        {
            tag_name = json_item[TAG_NAME].get<std::string>();

            // async, defer assignment with attribute addition/modification will be captured with attribute additions on nodes.
            if (tag_name == "SCRIPT" || tag_name == "script")
            {
                for (auto &json_attr : json_item["node_attributes"])
                {
                    if (json_attr["attr_name"] == "async")
                    {
                        async = true;
                    }
                    else if (json_attr["attr_name"] == "defer")
                    {
                        defer = true;
                    }
                }
            }
        }
        return std::make_tuple(tag_name, script_is_active, async, defer, attribute_name_and_values, previous_sibling_id);
    }

    std::tuple<std::string, std::string> JSONToAdGraphParser::ExtractJSONPropertiesAttributes(Utilities::ordered_json json_item)
    {

        std::string attr_name = "";
        std::string attr_value = "";

        attr_name = json_item["node_attribute"]["attr_name"].get<std::string>();
        attr_value = json_item["node_attribute"]["attr_value"].get<std::string>();

        return std::make_tuple(attr_name, attr_value);
    }

    std::tuple<bool, std::string, bool, std::string> JSONToAdGraphParser::ExtractJSONPropertiesForHTTPNode(Utilities::ordered_json json_item)
    {
        std::string active_script_id = json_item[ACTOR_ID].get<std::string>();
        bool script_is_active = (active_script_id == "0") ? false : true;

        std::string url = json_item[REQUEST_URL].get<std::string>();
        bool ad = (json_item[AD_CHECK].get<std::string>() == AD_TEXT) ? true : false;

        return std::make_tuple(script_is_active, url, ad, active_script_id);
    }

    // std::tuple<std::string, bool> JSONToAdGraphParser::ExtractJSONPropertiesForScriptNode(Utilities::ordered_json json_item){
    //     std::string script_text =  json_item["script_text"].get<std::string>();
    //     bool has_eval_or_function = false;
    //     int script_length = 0;

    //     script_length = script_text.length();

    //     std::size_t eval_found = script_text.find("eval");

    //     if (eval_found != std::string::npos) {
    //         has_eval_or_function = true;
    //     }
    //     else {
    //         std::size_t function_found = script_text.find("Function");
    //         if (function_found != std::string::npos)
    //         has_eval_or_function = true;
    //     }

    //     return std::make_tuple(script_length, has_eval_or_function);
    // }

    void JSONToAdGraphParser::CreateGraph(Utilities::ordered_json json_content)
    {
        //Timing
        Utilities::uint64 start_time = AdGraphAPI::Utilities::GetTimeMs64();

        int url_counter = 0;
        bool first_node_check = true;

        for (auto &json_item : json_content["timeline"])
        {

            // std::cout << json_item << std::endl;
            if (json_item[EVENT_TYPE] != SCRIPT_COMPILATION && json_item[EVENT_TYPE] != SCRIPT_EVAL && json_item[EVENT_TYPE] != NODE_ATTACH_LATER && json_item[EVENT_TYPE] != SCRIPT_EXECUTION && json_item[EVENT_TYPE] != SCRIPT_EXTENSION && adgraph_.CheckIfScriptIsNotAdded(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>()))
            {
                // std::cout << "\n\n HERE: " << SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>() << " \n\n";
                continue;
            }

            if (json_item[EVENT_TYPE] == NODE_INSERTION)
            {
                // Can have ACTOR_ID in addition to NODE_PARENT_ID;
                // add edge between parent and child along with actor id

                if (first_node_check)
                {
                    auto properties_tuple = ExtractJSONPropertiesForHTMLNode(json_item);
                    std::vector<std::string> temp_vector;
                    AdGraphAPI::HTMLNode *parent_node = adgraph_.CreateAndReturnHTMLNode(NODE_TEXT + json_item[NODE_PARENT_ID].get<std::string>(), false, "UNAVAILABLE", temp_vector, "0", false, false);
                    AdGraphAPI::HTMLNode *child_node = adgraph_.CreateAndReturnHTMLNode(NODE_TEXT + json_item[NODE_ID].get<std::string>(), std::get<1>(properties_tuple), std::get<0>(properties_tuple), std::get<4>(properties_tuple), std::get<5>(properties_tuple), std::get<2>(properties_tuple), std::get<3>(properties_tuple));

                    child_node->SetNodeInsertionWithScriptStatus(std::get<1>(properties_tuple));

                    parent_node->AddChild(child_node);
                    child_node->AddParent(parent_node);
                    adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                    first_node_check = false;
                    // See if we need to preserve child order.
                    // update_child_order(NodeChildOrder, json_item, NODE_PARENT_ID, NODE_ID);
                }
                else
                {
                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(NODE_TEXT + json_item[NODE_PARENT_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        AdGraphAPI::HTMLNode *child_node = static_cast<AdGraphAPI::HTMLNode *>(adgraph_.GetNode(NODE_TEXT + json_item[NODE_ID].get<std::string>()));
                        if (child_node != nullptr)
                        {
                            child_node->SetNodeInsertionWithScriptStatus((json_item[ACTOR_ID].get<std::string>() == "0") ? false : true);

                            parent_node->AddChild(child_node);
                            child_node->AddParent(parent_node);
                            adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                        else
                        {
                            auto properties_tuple = ExtractJSONPropertiesForHTMLNode(json_item);
                            child_node = adgraph_.CreateAndReturnHTMLNode(NODE_TEXT + json_item[NODE_ID].get<std::string>(), std::get<1>(properties_tuple), std::get<0>(properties_tuple), std::get<4>(properties_tuple), std::get<5>(properties_tuple), std::get<2>(properties_tuple), std::get<3>(properties_tuple));
                            child_node->SetNodeInsertionWithScriptStatus(std::get<1>(properties_tuple));

                            parent_node->AddChild(child_node);
                            child_node->AddParent(parent_node);
                            adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                    }
                }

                if (json_item[ACTOR_ID] != "0")
                {
                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());
                    if (parent_node != nullptr)
                    {
                        AdGraphAPI::HTMLNode *child_node = static_cast<AdGraphAPI::HTMLNode *>(adgraph_.GetNode(NODE_TEXT + json_item[NODE_ID].get<std::string>()));
                        if (child_node != nullptr)
                        {
                            child_node->SetNodeInsertionWithScriptStatus((json_item[ACTOR_ID].get<std::string>() == "0") ? false : true);

                            parent_node->AddChild(child_node);
                            child_node->AddParent(parent_node);
                            adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                        else
                        {
                            auto properties_tuple = ExtractJSONPropertiesForHTMLNode(json_item);
                            child_node = adgraph_.CreateAndReturnHTMLNode(NODE_TEXT + json_item[NODE_ID].get<std::string>(), std::get<1>(properties_tuple), std::get<0>(properties_tuple), std::get<4>(properties_tuple), std::get<5>(properties_tuple), std::get<2>(properties_tuple), std::get<3>(properties_tuple));
                            child_node->SetNodeInsertionWithScriptStatus(std::get<1>(properties_tuple));

                            parent_node->AddChild(child_node);
                            child_node->AddParent(parent_node);
                            adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }

            } // end NODE_INSERTION

            else if (json_item[EVENT_TYPE] == NODE_REMOVAL)
            {
                // Can only have ACTOR_ID;
                if (json_item[ACTOR_ID] != "0")
                {
                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        AdGraphAPI::HTMLNode *child_node = static_cast<AdGraphAPI::HTMLNode *>(adgraph_.GetNode(NODE_TEXT + json_item[NODE_ID].get<std::string>()));
                        if (child_node != nullptr)
                        {
                            child_node->SetNodeRemovalWithScriptStatus((json_item[ACTOR_ID].get<std::string>() == "0") ? false : true);
                            // fix issue that node/edge removal does not reduce node count
                            adgraph_.RemoveNode(child_node->GetId());
                            adgraph_.RemoveEdge(parent_node->GetId(), child_node->GetId());

                            // parent_node->AddChild(child_node);
                            // child_node->AddParent(parent_node);
                            //adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                        else
                        {
                            auto properties_tuple = ExtractJSONPropertiesForHTMLNode(json_item);
                            child_node = adgraph_.CreateAndReturnHTMLNode(NODE_TEXT + json_item[NODE_ID].get<std::string>(), std::get<1>(properties_tuple), std::get<0>(properties_tuple), std::get<4>(properties_tuple), std::get<5>(properties_tuple), std::get<2>(properties_tuple), std::get<3>(properties_tuple));
                            child_node->SetNodeRemovalWithScriptStatus(std::get<1>(properties_tuple));
                            adgraph_.RemoveNode(child_node->GetId());
                            adgraph_.RemoveEdge(parent_node->GetId(), child_node->GetId());

                            // parent_node->AddChild(child_node);
                            // child_node->AddParent(parent_node);
                            //adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                        }
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }

            } // end NODE_REMOVAL

            else if (json_item[EVENT_TYPE] == SCRIPT_COMPILATION)
            {
                AdGraphAPI::Node *parent_node = adgraph_.GetNode(NODE_TEXT + json_item[NODE_ID].get<std::string>());

                if (parent_node != nullptr)
                {
                    //check if it will be connected to a node or a script?
                    AdGraphAPI::HTTPNode *attachedHTTPNode = adgraph_.GetHTMLNodeToHTTPNodeMapping(static_cast<AdGraphAPI::HTMLNode *>(parent_node));

                    if (attachedHTTPNode != nullptr)
                    {
                        if (attachedHTTPNode->GetAd() == false)
                        {
                            // auto properties_tuple = ExtractJSONPropertiesForScriptNode(json_item);

                            AdGraphAPI::ScriptNode *child_node = adgraph_.CreateAndReturnScriptNode(SCRIPT_TEXT + json_item[SCRIPT_ID].get<std::string>(), json_item["script_text"].get<std::string>(), false);
                            attachedHTTPNode->AddChild(child_node);
                            child_node->AddParent(attachedHTTPNode);
                            adgraph_.AddEdge(attachedHTTPNode->GetId(), child_node->GetId());
                        }
                        else
                        {
                            adgraph_.AddNotAddedScript(SCRIPT_TEXT + json_item[SCRIPT_ID].get<std::string>());
                        }
                    }
                    else
                    {
                        // auto properties_tuple = ExtractJSONPropertiesForScriptNode(json_item);

                        AdGraphAPI::ScriptNode *child_node = adgraph_.CreateAndReturnScriptNode(SCRIPT_TEXT + json_item[SCRIPT_ID].get<std::string>(), json_item["script_text"].get<std::string>(), false);
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                    }
                }
            } // end SCRIPT_COMPILATION

            else if (json_item[EVENT_TYPE] == SCRIPT_EVAL)
            {
                AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[SCRIPT_PARENT_ID].get<std::string>());
                if (parent_node != nullptr)
                {
                    // auto properties_tuple = ExtractJSONPropertiesForScriptNode(json_item);

                    AdGraphAPI::ScriptNode *child_node = adgraph_.CreateAndReturnScriptNode(SCRIPT_TEXT + json_item[SCRIPT_ID].get<std::string>(), json_item["script_text"].get<std::string>(), true);
                    parent_node->AddChild(child_node);
                    child_node->AddParent(parent_node);
                    adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());
                }
            } // end SCRIPT_EVAL

            else if (json_item[EVENT_TYPE] == ATTR_ADDITION || json_item[EVENT_TYPE] == ATTR_MODIFICATION || json_item[EVENT_TYPE] == ATTR_REMOVAL || json_item[EVENT_TYPE] == ATTR_STYLE_TEXT_ADDITION || json_item[EVENT_TYPE] == ATTR_STYLE_REMOVAL)
            {
                // (TODO:) what if we do not have any script actor? Possible options. 1) ignore?
                // (NOTE:) For actor_id check script_actor(true/false) as well.
                if (json_item[ACTOR_ID] != "0")
                {

                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());
                    if (parent_node != nullptr)
                    {
                        AdGraphAPI::HTMLNode *child_node = static_cast<AdGraphAPI::HTMLNode *>(adgraph_.GetNode(NODE_TEXT + json_item[NODE_ID].get<std::string>()));
                        if (child_node != nullptr)
                        {
                            parent_node->AddChild(child_node);
                            child_node->AddParent(parent_node);
                            adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                            auto properties_tuple = ExtractJSONPropertiesAttributes(json_item);
                            child_node->AddAttributeNameAndValue(std::get<0>(properties_tuple));
                            child_node->AddAttributeNameAndValue(std::get<1>(properties_tuple));

                            if (json_item[EVENT_TYPE] == ATTR_ADDITION)
                                child_node->SetAttributeAdditionWithScriptStatus(true);
                            else if (json_item[EVENT_TYPE] == ATTR_MODIFICATION)
                                child_node->SetAttributeModificationWithScriptStatus(true);
                            else if (json_item[EVENT_TYPE] == ATTR_REMOVAL)
                                child_node->SetAttributeRemovalWithScriptStatus(true);
                            else if (json_item[EVENT_TYPE] == ATTR_STYLE_TEXT_ADDITION)
                                child_node->SetAttributeStyleAdditionWithScriptStatus(true);
                            else if (json_item[EVENT_TYPE] == ATTR_STYLE_REMOVAL)
                                child_node->SetAttributeStyleRemovalWithScriptStatus(true);
                        }
                        else
                        {
                            // Defer edge addition concept: keep edges for nodes not yet inserted into the DOM.
                            // When inserted see if any pending connections need to be added.
                        }
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }
            } // end ATTR manipulations

            else if (json_item[EVENT_TYPE] == NODE_ATTACH_LATER)
            {
                // keep attach later in a list and do not add to actual graph (the edge will be added in node_insertion event).
                // only add attach later event if there is no script_actor then erase it. If there is a script actor erase the attach later event any way.
                // one connection will be added in both of the cases.

                adgraph_.AddAttachLaterEvent(json_item[NODE_ID], json_item[NODE_PARENT_ID]);
            } // end NODE_ATTACH_LATER

            else if ((json_item[EVENT_TYPE] == NETWORK_IFRAME || json_item[EVENT_TYPE] == NETWORK_LINK) && json_item[REQUEST_URL] != "")
            {
                url_counter += 1;
                bool classify = false;
                AdGraphAPI::HTTPNode *child_node_for_classification;

                if (json_item[ACTOR_ID] != "0")
                {

                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());
                    if (parent_node != nullptr)
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                } // END Actor_id if stattement

                AdGraphAPI::Node *parent_node = adgraph_.GetNode(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());
                if (parent_node != nullptr)
                {
                    AdGraphAPI::HTTPNode *child_node = static_cast<AdGraphAPI::HTTPNode *>(adgraph_.GetNode(URL_TEXT + std::to_string(url_counter)));
                    if (child_node != nullptr)
                    {
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                    else
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                }

                if (classify)
                {
                    Utilities::WriteFeatures(adgraph_.GetAllProperties(child_node_for_classification, json_item[EVENT_TYPE].get<std::string>()), adgraph_.GetBaseDomain(), features_file_name_);
                    Utilities::WriteURLIdStringMapping(URL_TEXT + std::to_string(url_counter), child_node_for_classification->GetURL(), url_id_string_map_file_name_);
                }

            } // end NETWORK_IFRAME, NETWORK_LINK

            else if (json_item[EVENT_TYPE] == NETWORK_XMLHTTP && json_item[REQUEST_URL] != "")
            {
                url_counter += 1;
                if (json_item[ACTOR_ID] != "0")
                {
                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);

                        AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        Utilities::WriteFeatures(adgraph_.GetAllProperties(child_node, json_item[EVENT_TYPE].get<std::string>()), adgraph_.GetBaseDomain(), features_file_name_);
                        Utilities::WriteURLIdStringMapping(URL_TEXT + std::to_string(url_counter), child_node->GetURL(), url_id_string_map_file_name_);
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }
            } // end NETWORK_XMLHTTP

            else if (json_item[EVENT_TYPE] == NETWORK_SCRIPT && json_item[REQUEST_URL] != "")
            {
                url_counter += 1;
                bool classify = false;
                AdGraphAPI::HTTPNode *child_node_for_classification;

                if (json_item[ACTOR_ID] != "0")
                {

                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }

                AdGraphAPI::Node *parent_node = adgraph_.GetNode(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());
                if (parent_node != nullptr)
                {
                    AdGraphAPI::HTTPNode *child_node = static_cast<AdGraphAPI::HTTPNode *>(adgraph_.GetNode(URL_TEXT + std::to_string(url_counter)));
                    if (child_node != nullptr)
                    {
                        adgraph_.AddHTMLNodeToHTTPNodeMapping(static_cast<AdGraphAPI::HTMLNode *>(parent_node), child_node);
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                    else
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                        adgraph_.AddHTMLNodeToHTTPNodeMapping(static_cast<AdGraphAPI::HTMLNode *>(parent_node), child_node);
                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                }

                if (classify)
                {
                    Utilities::WriteFeatures(adgraph_.GetAllProperties(child_node_for_classification, json_item[EVENT_TYPE].get<std::string>()), adgraph_.GetBaseDomain(), features_file_name_);
                    Utilities::WriteURLIdStringMapping(URL_TEXT + std::to_string(url_counter), child_node_for_classification->GetURL(), url_id_string_map_file_name_);
                }
            } // end NETWORK_SCRIPT

            else if ((json_item[EVENT_TYPE] == NETWORK_IMAGE || json_item[EVENT_TYPE] == NETWORK_VIDEO) && json_item[REQUEST_URL] != "")
            {
                url_counter += 1;
                bool classify = false;
                AdGraphAPI::HTTPNode *child_node_for_classification;

                if (json_item[ACTOR_ID] != "0")
                {

                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(SCRIPT_TEXT + json_item[ACTOR_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;

                        adgraph_.RemoveAttachLaterEvent(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());
                    }
                    else
                    {
                        // (TODO) UpdateUnseenAndFrameScriptCount
                    }
                }
                else
                {
                    AdGraphAPI::Node *parent_node = adgraph_.GetNode(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());

                    if (parent_node != nullptr)
                    {
                        auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                        AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                        parent_node->AddChild(child_node);
                        child_node->AddParent(parent_node);
                        adgraph_.AddEdge(parent_node->GetId(), child_node->GetId());

                        classify = true;
                        child_node_for_classification = child_node;
                    }
                    else
                    {
                        AdGraphAPI::Node *parent_node = adgraph_.GetAttachLaterParentNode(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());

                        if (parent_node != nullptr)
                        {
                            AdGraphAPI::HTMLNode *child_node_attachlater = static_cast<AdGraphAPI::HTMLNode *>(adgraph_.GetNode(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>()));
                            if (child_node_attachlater != nullptr)
                            {
                                parent_node->AddChild(child_node_attachlater);
                                child_node_attachlater->AddParent(parent_node);
                                adgraph_.AddEdge(parent_node->GetId(), child_node_attachlater->GetId());

                                auto properties_tuple = ExtractJSONPropertiesForHTTPNode(json_item);
                                AdGraphAPI::HTTPNode *child_node = adgraph_.CreateAndReturnHTTPNode(URL_TEXT + std::to_string(url_counter), std::get<0>(properties_tuple), std::get<3>(properties_tuple), std::get<1>(properties_tuple), std::get<2>(properties_tuple));

                                child_node_attachlater->AddChild(child_node);
                                child_node->AddParent(child_node_attachlater);
                                adgraph_.AddEdge(child_node_attachlater->GetId(), child_node->GetId());

                                classify = true;
                                child_node_for_classification = child_node;
                            }
                            adgraph_.RemoveAttachLaterEvent(NODE_TEXT + json_item[REQUESTOR_ID].get<std::string>());
                        }
                    }
                }
                if (classify)
                {
                    Utilities::WriteFeatures(adgraph_.GetAllProperties(child_node_for_classification, json_item[EVENT_TYPE].get<std::string>()), adgraph_.GetBaseDomain(), features_file_name_);
                    Utilities::WriteURLIdStringMapping(URL_TEXT + std::to_string(url_counter), child_node_for_classification->GetURL(), url_id_string_map_file_name_);
                }
            } // end NETWORK_IMAGE, NETWORK_VIDEO
        }     // end json_content loop

        //Timing
        Utilities::uint64 end_time = AdGraphAPI::Utilities::GetTimeMs64();
        Utilities::ordered_json json_timing;
        Utilities::ordered_json overall_timing;
        overall_timing["overall_time"] = end_time - start_time;
        Utilities::ordered_json properties_timing = adgraph_.GetTimingInfo();

        json_timing.update(overall_timing);
        json_timing.update(properties_timing);
        Utilities::WriteTimingInfo(timing_file_name_, adgraph_.GetBaseDomain(), json_timing);

        Utilities::ordered_json json_visualization = adgraph_.PrepareJSONVisualization();
        Utilities::WriteJSON(visualization_file_name_, json_visualization);
    } // end CreateGraph function

} // namespace AdGraphAPI

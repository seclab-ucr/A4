#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import json, argparse, codecs, HTMLParser, ftfy, itertools
from copy import copy, deepcopy
import networkx as nx
from lxml import html
from bs4 import BeautifulSoup, Tag, Comment
from collections import OrderedDict, defaultdict
from operator import itemgetter

SCRIPT_COMPILATION = "ScriptCompilation"
SCRIPT_EXECUTION = "ScriptExecution"

NETWORK_IFRAME = "NetworkIframeRequest"
NETWORK_SCRIPT = "NetworkScriptRequest"
NETWORK_IMAGE = "NetworkImageRequest"
NETWORK_LINK = "NetworkLinkRequest"
NETWORK_XMLHTTP = "NetworkXMLHTTPRequest"

NODE_ID = "node_id"
ACTOR_ID = "actor_id"
SCRIPT_ID = "script_id"
REQUESTOR_ID = "requestor_id"
NODE_PARENT_ID = "node_parent_id"
NODE_PREVIOUS_ID = "node_previous_sibling_id"
NODE_TYPE = "node_type"
TAG_NAME = "tag_name"

EVENT_TYPE = "event_type"

NODE_TEXT = "NODE_"
SCRIPT_TEXT = "SCRIPT_"

NODE_CREATION = "NodeCreation"
NODE_INSERTION = "NodeInsertion"
NODE_REMOVAL = "NodeRemoval"
ATTR_ADDITION = "AttrAddition"
ATTR_MODIFICATION = "AttrModification"
ATTR_REMOVAL = "AttrRemoval"
ATTR_STYLE_ADDITION = "AttrStyleAddition"
ATTR_STYLE_TEXT_ADDITION = "AttrStyleTextAddition"
ATTR_STYLE_REMOVAL = "AttrStyleRemoval"

print_mode = True

def printing(p, what):
	if p:
		print(what)

def areTwoDictListEqual(soup_style_dicts, graph_style_dicts):
	equality = False
	if len(soup_style_dicts) == len(graph_style_dicts):
		if len(soup_style_dicts) == 1 and soup_style_dicts[0]['style_name'] == graph_style_dicts[0]['style_name'] and soup_style_dicts[0]['style_value'] == graph_style_dicts[0]['style_value']:
			equality = True 
		else:
			s = [sorted(l, key=itemgetter('style_name')) for l in (soup_style_dicts, graph_style_dicts)]
			pairs = zip(s[0], s[1])
			key_diffs = [[style_key for style_key in x if x[style_key] != y[style_key]] for x, y in pairs if x != y]
			if not key_diffs:
				equality = True
	return equality

def have_same_attr(soup_tag, graph_node):
	ignore_style = True
	equality = False
	if 'node_attributes' in graph_node:
		if len(soup_tag.attrs) == 0 and ('node_attributes' not in graph_node or ('node_attributes' in graph_node and len(graph_node['node_attributes']) == 0)):
			equality = True
		else:
			for attr_name, value in soup_tag.attrs.iteritems():
				if type(value) is list:
					attr_value = " ".join(value).strip()
				else:
					attr_value = "".join(value).strip()
				if attr_name.strip() == 'style':
					# create a list of style dicts
					soup_style_dicts = create_style_dict(attr_value) 
					if any(_['attr_name'] == 'style' for _ in graph_node['node_attributes']):
						# get list of style dicts
						graph_style_dicts = [_['attr_value'] for _ in graph_node['node_attributes'] if _['attr_name'] == 'style'][0]
						# iterate through zip of list of style dicts that are sorted by style_name
						if areTwoDictListEqual(soup_style_dicts, graph_style_dicts):
							equality = True
							break
						elif ignore_style:
							equality = True
							break
						else:
							printing(p=print_mode, what="[Attribute Discrepancy] \n --[Soup (%s)] attribute name: [%s], attribute value: [%s] \n --[Graph (%s)] attribute: %s"%(soup_tag.name, attr_name, attr_value, graph_node['node_id'], str(graph_node['node_attributes'])))
							equality = False
							break
				elif any(_['attr_name'] == attr_name and _['attr_value'] == attr_value for _ in graph_node['node_attributes']):
					equality = True
				else:
					printing(p=print_mode, what="[Attribute Discrepancy] \n --[Soup (%s)] attribute name: [%s], attribute value: [%s] \n --[Graph (%s)] attribute: %s"%(soup_tag.name, attr_name, attr_value, graph_node['node_id'], str(graph_node['node_attributes'])))
					equality = False
					break					
	elif (('node_attributes' in graph_node and len(graph_node['node_attributes']) == 0) or 'node_attributes' not in graph_node) and len(soup_tag.attrs) == 0:
		equality = True
	if not equality and not ignore_style:
		printing(p=print_mode, what="[Attribute Discrepancy] \n --[Soup (%s)] attributes: [%s] \n --[Graph (%s)] attribute: %s"%(soup_tag.name, str(soup_tag.attrs), graph_node['node_id'], str(graph_node['node_attributes']) if 'node_attributes' in graph_node else None))
		printing(p=print_mode, what='-'*80)
	return equality

def have_same_children(graph, soup_tag, graph_node, graph_node_id):
	equality = False
	children_soup = [_ for _ in soup_tag.children if isinstance(_, Tag)]
	# ex. graph_node -> G['20']
	children_graph = [_ for _ in graph[graph_node_id]]
	if len(children_soup) == len(children_graph):
		if [child.name.lower() for child in children_soup] == [graph.nodes[child]['tag_name'].lower() for child in children_graph]:
			equal_child_counter = 0
			for pair in itertools.izip_longest(children_soup, children_graph):
				s_child = pair[0]
				g_child = pair[1]
				if s_child.name.lower() == graph.nodes[g_child]['tag_name'].lower() and have_same_attr(s_child, graph.nodes[g_child]):
					equal_child_counter += 1
				else:
					printing(p=print_mode, what="[Children Discrepancy] [Attributes are not the same] \n --[Soup (%s)] \n --[Graph (%s)] %s"%(s_child.name, graph.nodes[g_child]['node_id'], graph.nodes[g_child]['tag_name'].lower()))
	
			if len(children_soup) == len(children_graph) == equal_child_counter:
				equality = True
			else:
				printing(p=print_mode, what="[Children Discrepancy] Length is not equal: [Soup (%s)], [Graph (%s)] %s:"%(soup_tag.name, graph_node['node_id'], graph.nodes[graph_node_id]['tag_name'].lower()))

	if not equality:
		print("Graph %s"%[graph.node[_]['node_id'] for _ in children_graph])
		print("Soup %s"%[_.name for _ in children_soup])
		printing(p=print_mode, what="[Children Discrepancy] [Soup (%s)] [Graph (%s)] %s"%(soup_tag.name, graph_node['node_id'], graph_node['tag_name']))
		for c in itertools.izip_longest([_.name for _ in children_soup],[(graph.node[_]['node_id'], graph.node[_]['tag_name']) for _ in children_graph if 'tag_name' in graph.node[_]]):
			#if c[0].lower() != c[1][1].lower():
			if c[0] and c[1]:
				printing(p=print_mode, what="--[child] [Soup] %s, [Graph (%s)] %s"%(c[0], c[1][0], c[1][1]))
			# elif child is None 
			else:
				printing(p=print_mode, what="--[child] [Soup] %s, [Graph (%s)] [None child]"%(c[0], c[1]))
		printing(p=print_mode, what='-'*80)
	return equality

def have_same_text(soup_tag, graph_node, node_id):
	equality = False
	if 'textContent' in graph_node and " ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip() != None: #soup_tag.find(text=True, recursive=False) != None:
		#if " ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip().split() == graph_node['textContent'].strip().encode('latin1').decode('utf8').split():
		if html_parser.unescape(" ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip()).split() == ftfy.fix_text_encoding(graph_node['textContent'].strip()).split():
			equality = True
		else:
			printing(p=print_mode, what="[TEXT] \n --[Soup (%s)]: text: %s \n --[Graph (%s)]: text: %s"%(soup_tag.name, html_parser.unescape(" ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip()), graph_node['tag_name'], graph_node['textContent']))
	elif 'textContent' not in graph_node:
		if soup_tag.find(text=True, recursive=False) is None: 
			equality = True
		elif not soup_tag.find(text=True, recursive=False).strip():
			equality = True
	if not equality:
		printing(p=print_mode, what="[TEXT] \n --[Soup (%s)]: text: %s \n --[Graph (%s)]: text: %s"%(soup_tag.name, html_parser.unescape(" ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip()) if " ".join(map(lambda _:_.strip(), soup_tag.findAll(text=True, recursive=False))).strip() else None, graph_node['node_id'], ftfy.fix_encoding(graph_node['textContent']) if 'textContent' in graph_node else None))
		printing(p=print_mode, what='-'*80)
	return equality
		
def is_equal(graph, soup_tag, graph_node, graph_node_id):
	equality = False
	try:
		if 'tag_name' in graph_node and soup_tag.name:
			if soup_tag.name.lower() == graph_node['tag_name'].lower():
				#if have_same_text(soup_tag, graph_node, graph_node_id):
				if have_same_attr(soup_tag, graph_node): 
					if have_same_children(graph, soup_tag, graph_node, graph_node_id):
						equality = True
					else:
						printing(p=print_mode, what="[Not Equal Children] [Soup]: %s, [Graph (%s)]: %s"%(soup_tag.name, graph_node['node_id'], graph_node['tag_name']))
						#printing(print_mode=print_mode, soup_tag.name, soup_tag.attrs, graph_node['tag_name'], graph_node['node_attributes'] if 'node_attributes' in graph_node else None)
				else:
					printing(p=print_mode, what="[Not Equal Attribute] [Soup]: %s, [Graph (%s)]: %s"%(soup_tag.name, graph_node['node_id'],  graph_node['tag_name']))
				#else:
				#	printing(p=print_mode, what="[Not Equal TEXT] [Soup]: %s, [Graph (%s)]: %s"%(soup_tag.name,  graph_node['node_id'], graph_node['tag_name']))
			else:
				printing(p=print_mode, what="[Not Equal TagName] [Soup]: %s, [Graph (%s)]: %s"%(soup_tag.name, graph_node['node_id'], graph_node['tag_name']))
	except:
		print(soup_tag)
	return equality

def compare_graph_soup(graph, soup):
	print_only_tags = False
	printing(p=print_mode, what="Comparing graph and soup")
	diff = list()
	count = 0
	# remove all comments
	comments = soup.findAll(text=lambda text:isinstance(text, Comment))
	for c in comments:
		c.extract()

	printing(p=print_mode, what="length DOM: %d"%(len([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)])))
	printing(p=print_mode, what="length Graph: %d"%(len([_ for _ in G])))
	
	#for pair in itertools.izip_longest([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag) and _.name.lower() != 'noscript'], [_ for _ in G if 'tag_name' in G.node[_] and G.node[_]['tag_name'].lower() != 'noscript']):
	for pair in itertools.izip_longest([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)], [_ for _ in G if 'tag_name' in G.node[_]]):
	#for pair in itertools.izip_longest([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)], [_ for _ in G]):
		if print_only_tags:
			if pair[0].name.lower() != graph.node[pair[1]]['tag_name'].lower():
				printing(p=print_mode, what="%s, %s, %s"%(pair[0].name, graph.node[pair[1]]['tag_name'], graph.node[pair[1]]['node_id']))
		else:
			# if any(not _ for _ in (pair[0], )
			if is_equal(graph, pair[0], graph.node[pair[1]], pair[1]):
				printing(p=print_mode, what="%s, %s"%(pair[0].name, graph.node[pair[1]]['tag_name']))
				count += 1
	if not print_only_tags:			
		if count == len([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)]):
			print("equal", count, len([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)]))
		else:
			print("not equal", count, len([_ for _ in soup.recursiveChildGenerator() if isinstance(_, Tag)]))

def build_graph(timeline):
	G = nx.OrderedDiGraph()
	G_1 = nx.OrderedDiGraph()
	should_get_removed, should_removed_text_nodes = set(), set()
	parent_child_ordered_dict = OrderedDict() 
	doctype_id = None
	#create Doctype
	if 'node_type' in timeline[0] and str(timeline[0]['node_type']) == "10":
		doctype_id = str(timeline[0]['node_parent_id'])
		G.add_node(doctype_id)
		parent_child_ordered_dict[doctype_id] = []
	for _ in timeline:
		if 'tag_name' in _ and _['tag_name'].lower() == 'html' and 'node_type' in _ and str(_['node_type']) == "1":
			html_node = _['node_id']
			break
	for event in timeline:
		if 'node_id' in event:
			node_id = str(event['node_id'])
			if 'node_parent_id' in event and event['node_parent_id'] != 0:
				node_parent_id = str(event['node_parent_id'])
			################### NODE CREATION
			if event['event_type'].lower() == NODE_CREATION.lower():
				attributes = dict()
				for key in ['node_parent_id', 'node_previous_sibling_id', 'node_type']:
					if  key in event:
						if key == 'node_type':
							attributes[key] = str(event[key])
						else:
							attributes[key] = event[key]
				if 'node_attributes' in event:
					attributes['node_attributes'] = [{'attr_name':attr['attr_name'].strip(), 'attr_value':attr['attr_value'].strip()} for attr in event['node_attributes']]

				if 'tag_name' in event:
					if str(event['node_type']) == '3':
						if event['tag_name']:
							attributes['tag_name'] = 'TEXT'
							#attributes['textContent'] = event['tag_name'].replace(u'\xc3\xa0', ' ').replace(u'\xc2\xa0', ' ')
							attributes['textContent'] = ftfy.fix_encoding(event['tag_name'].replace(u'\xc3\xa0', ' ').replace(u'\xc2\xa0', ' ').strip())
						should_get_removed.update([str(event['node_id'])])  
					else:
						attributes['tag_name'] = event['tag_name']
				G.add_node(node_id)
				for attr in attributes:
					G.node[node_id][attr] = attributes[attr]

			################### NODE INSERTION 
			if event['event_type'].lower() == NODE_INSERTION.lower():
				if 'node_type' in event and str(event['node_type']) == "3":
					#printing(print_mode=print_mode, node_parent_id, node_id)
					#if node_id not in G:
					#G.add_node(node_id)
					if node_parent_id not in parent_child_ordered_dict:
						parent_child_ordered_dict[node_parent_id] = []
					if 'node_previous_sibling_id' in event:
						if str(event['node_previous_sibling_id']) == "0":
							parent_child_ordered_dict[node_parent_id].insert(0, node_id)
						elif str(event['node_previous_sibling_id']) in parent_child_ordered_dict[node_parent_id]:
						#else:
							parent_child_ordered_dict[node_parent_id].insert(parent_child_ordered_dict[node_parent_id].index(str(event['node_previous_sibling_id'])) + 1, node_id)
					should_removed_text_nodes.update([node_id])
					if node_parent_id in G and 'textContent' in G.node[node_id]:
						if 'textContent' in G.node[node_parent_id] and (('script_actor' in event and event['script_actor'] is False) or 'script_actor' not in event): 
							#if G.node[node_parent_id]['tag_name'] == 'SCRIPT':
								#printing(print_mode=print_mode, "before", node_parent_id, node_id,  G.node[node_parent_id]['textContent'])
								#printing(print_mode=print_mode, event['script_actor'])
							G.node[node_parent_id]['textContent'] += ' %s'%(G.node[node_id]['textContent'].replace(u'\xc3\xa0', ' ').replace(u'\xc2\xa0', ' ').strip())
							#if G.node[node_parent_id]['tag_name'] == 'SCRIPT':
								#printing(print_mode=print_mode, "after", node_parent_id, node_id, G.node[node_parent_id]['textContent'])
								#printing(print_mode=print_mode, event['script_actor'])
						else:
							G.node[node_parent_id]['textContent'] = G.node[node_id]['textContent'].replace(u'\xc3\xa0', ' ').replace(u'\xc2\xa0', ' ').strip()

				else:
					attributes = defaultdict(list)
					for key in ['node_parent_id', 'node_previous_sibling_id', 'node_type', 'tag_name']:
						if  key in event:
							# remove comments
							if str(event['node_type']) == '8':
								should_removed_text_nodes.update([str(event['node_id'])])
							elif key == 'node_type':
								attributes[key] = str(event[key])
							else:
								attributes[key] = event[key]
					if 'node_attributes' in event:
						for attr in event['node_attributes']:
							if attr['attr_name'] == 'style':
								attributes['node_attributes'].append({'attr_name': 'style', 'attr_value': create_style_dict(attr['attr_value'])})
							else:
								attributes['node_attributes'].append({'attr_name':attr['attr_name'].encode('latin1').decode('utf8').strip(), 'attr_value':attr['attr_value'].encode('latin1').decode('utf8').strip()})

					if node_parent_id in G:
						if node_id not in G:
							G.add_node(node_id)
						for attr in attributes:
							G.node[node_id][attr] = attributes[attr]
						## insert here
						if node_parent_id not in parent_child_ordered_dict:
							parent_child_ordered_dict[node_parent_id] = []
						if 'node_previous_sibling_id' in event:
							if str(event['node_previous_sibling_id']) == "0":
								parent_child_ordered_dict[node_parent_id].insert(0, node_id)
							else:
								parent_child_ordered_dict[node_parent_id].insert(parent_child_ordered_dict[node_parent_id].index(str(event['node_previous_sibling_id'])) + 1, node_id)
							if node_id in should_get_removed:
								should_get_removed.remove(node_id)
					#else:
					#    printing(print_mode=print_mode, "This node is orphan! --> %s"%(node_id))

			################### NODE REMOVAL 
			if event['event_type'].lower() == NODE_REMOVAL.lower():
				#if node_id in G:
				should_get_removed.update([node_id])
				for i in parent_child_ordered_dict:
					if node_id in parent_child_ordered_dict[i]:
						parent_child_ordered_dict[i].remove(node_id)
					#G.remove_node(node_id)
			################### ATTRIBUTE ADDITION/MOD/REMOVE 
			if any(event['event_type'].lower() == _.lower() for _ in (ATTR_ADDITION, ATTR_MODIFICATION, ATTR_REMOVAL, ATTR_STYLE_ADDITION, ATTR_STYLE_REMOVAL, ATTR_STYLE_TEXT_ADDITION)):
				if node_id in G:
					attribute_list = None
					if event['event_type'] == ATTR_ADDITION:
						if 'node_attributes' not in G.node[node_id]:
							G.node[node_id]['node_attributes'] = []
						if type(event['node_attribute']) is not list:
							attribute_list = [event['node_attribute']]
						else:
							attribute_list = event['node_attribute']
						graph_attrib_list = list(G.node[node_id]['node_attributes'])
						for attr in attribute_list:
							if graph_attrib_list != []:
								for item in copy(G.node[node_id]['node_attributes']):
									if item['attr_name'] == attr['attr_name'].strip():
										#style --> {'attr_name': 'style', 'attr_value': {'style_name': '..', 'style_value': '..' }}
										if item['attr_name'] == 'style':
											style_dict = create_style_dict(attr['attr_value'])
											temp_style_list = list()
											for node_style_attr in item['attr_value']:
												for event_style_attr in style_dict:
													if node_style_attr['style_name'] == event_style_attr['attr_name'].strip() and node_style_attr['style_value'] != event_style_attr['attr_value'].strip():
														temp_style_list.remove({'style_name': node_style_attr['style_name'], 'style_value': node_style_attr['style_value']})
														temp_style_list.append({'style_name': event_style_attr['attr_name'].strip(), 'style_value': event_style_attr['attr_value'].strip()})
											graph_attrib_list.remove({'attr_name': 'style', 'attr_value': item['attr_value']})
											graph_attrib_list.append({'attr_name': 'style', 'attr_value': temp_style_list})

										elif item['attr_value'] != attr['attr_value'].strip():
											graph_attrib_list.remove({'attr_name': item['attr_name'], 'attr_value': item['attr_value']})
											graph_attrib_list.append({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})

									elif {'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()} not in graph_attrib_list:
										if attr['attr_name'] == 'style':
											style_dict = create_style_dict(attr['attr_value'])
											graph_attrib_list.append({'attr_name': 'style', 'attr_value': style_dict})
										else:
											graph_attrib_list.append({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})
							else:
								if attr['attr_name'] == 'style':
									style_dict = create_style_dict(attr['attr_value'])
									graph_attrib_list.append({'attr_name': 'style', 'attr_value': style_dict})
								else:
									graph_attrib_list.append({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})
								#G.node[node_id]['node_attributes'].append({'attr_name': 'style', 'attr_value': [{'style_name': attr['attr_name'], 'style_value': attr['attr_value']}]})
						G.node[node_id]['node_attributes'] = graph_attrib_list
					elif 'node_attributes' in G.node[node_id] and G.node[node_id]['node_attributes']:
						if type(event['node_attribute']) is not list:
							attribute_list = [event['node_attribute']]
						else:
							attribute_list = event['node_attribute']
						graph_attrib_list = list(G.node[node_id]['node_attributes'])
						for attr in attribute_list:
							for pre_attr in graph_attrib_list:
								# document.querySelector().style = "display: none"
								#if event['event_type'] == ATTR_STYLE_TEXT_ADDITION:

								# document.querySelector().style.display = "none"
								if any(event['event_type'] == _ for _ in (ATTR_STYLE_ADDITION, ATTR_STYLE_REMOVAL)):
									if pre_attr['attr_name'] == 'style': 
										# {'attr_name': 'style', 'attr_value': [{'style_name': position', 'style_value':'..', {'style_name': 'display', 'style_value': '..', ...}]}
										for style_pair in pre_attr['attr_value']:
											if style_pair['style_name'] == attr['attr_name'].strip():
												if event['event_type'] == ATTR_STYLE_ADDITION:
													style_pair['style_value'] = attr['attr_value'].strip()
												elif event['event_type'] == ATTR_STYLE_REMOVAL and style_pair['style_value'] == attr['attr_value'].strip():
													pre_attr['attr_value'].remove(style_pair)
										#pre_attr['attr_value'] = pre_attr['attr_value'] + ' ' + attr['attr_name'] + ': ' + attr['attr_value'] + ';' 
									else:
										style_dict = create_style_dict(attr['attr_name'].strip() + ': ' + attr['attr_value'].strip() + ';')
										G.node[node_id]['node_attributes'].append({'attr_name': 'style', 'attr_value': style_dict})
								elif event['event_type'] == ATTR_STYLE_TEXT_ADDITION:
									if pre_attr['attr_name'] == 'style':
										style_dict = create_style_dict(attr['attr_value'])
										for style_pair in pre_attr['attr_value']:
											for style_item in style_dict:
												if style_pair['style_name'] == style_item['style_name'] and style_pair['style_value'] != style_item['style_value']:
													style_pair['style_value'] = style_item['style_value']

								elif event['event_type'] == ATTR_REMOVAL:
									if pre_attr['attr_name'] == attr['attr_name'].strip() and pre_attr['attr_value'] == attr['attr_value'].strip():
										G.node[node_id]['node_attributes'].remove({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})
								elif event['event_type'] == ATTR_MODIFICATION:
									if pre_attr['attr_name'] == attr['attr_name'].strip() and pre_attr['attr_value'] != attr['attr_value'].strip():
										G.node[node_id]['node_attributes'].remove({'attr_name': pre_attr['attr_name'], 'attr_value': pre_attr['attr_value']})
										G.node[node_id]['node_attributes'].append({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})
									
								else:
									if pre_attr['attr_name'] == attr['attr_name'].strip() and pre_attr['attr_value'] != attr['attr_value'].strip():
										#update value
										pre_attr['attr_value'] = attr['attr_value'].strip()
									else:
										G.node[node_id]['node_attributes'].append({'attr_name': attr['attr_name'].strip(), 'attr_value': attr['attr_value'].strip()})


				else:
					printing(p=print_mode, what="this never should be called")
					G.add_node(node_id, node_attributes=[{'attr_name': event['node_attribute']['attr_name'], 'attr_value': event['node_attribute']['attr_value']}])
	for parent in G:
		if parent in parent_child_ordered_dict:
			G.add_edges_from([(parent, child) for child in parent_child_ordered_dict[parent] if child not in should_removed_text_nodes])
		
	printing(p=print_mode, what="removing nodes")
	if doctype_id:
		G.remove_node(doctype_id)
	
	for i in G.nodes():
		if not G.pred[i] and i != html_node:
			should_get_removed.update([i])
	G.remove_nodes_from(list(should_get_removed))
	G.remove_nodes_from(list(nx.isolates(G)))
	
	global_counter = int(html_node)
	gnode_to_counter_map = dict()
	for node in dfs(parent_child_ordered_dict, html_node):
		if node in G:
			gnode_to_counter_map[node] = global_counter
			G_1.add_node(global_counter)
			for item in G.node[node]:
				G_1.node[global_counter][item] = deepcopy(G.node[node][item])
				G_1.node[global_counter]['node_id'] = node
			
			if G.pred[node].keys() and G.pred[node].keys()[0] in gnode_to_counter_map:
				G_1.add_edge(gnode_to_counter_map[G.pred[node].keys()[0]], global_counter)
			global_counter +=1

	return G_1	

def dfs(graph, start):
	visited, stack = list(), [start]
	while stack:
		vertex = stack[0]
		del stack[0]
		if vertex not in visited:
			visited.append(vertex)
			if vertex in graph:
				stack = graph[vertex] + stack
	return visited

def search_graph(graph, element_name, element_attributes):
	#element_attributes = [{'attr_name': name, 'attr_value': value}, ...]
	for child in G:
		if G.node[child]['tag_name'].lower() == element_name.lower():
			child_attrib = [_ for _ in G.node[child]['node_attributes'] if 'node_attributes' in G.node[child]]
			if all((pair[0]['attr_name'], pair[0]['attr_value']) == (pair[1]['attr_name'], pair[1]['attr_value']) for pair in zip(element_attributes, child_attrib)):
				printing(p=print_mode, what=G.node[child])

def create_style_dict(style_string):
	#border: none; padding: 0;
	if len(style_string.split(';')) > 2:
		style_dict = [{'style_name': pair.split(':')[0].strip(),'style_value': pair.split(':')[1].strip().replace(';', '')} for pair in style_string.split(';') if pair]
	elif style_string == '':
		style_dict = [{'style_name': 'style', 'style_value': ''}]
	else:
		style_dict = [{'style_name': style_string.split(':')[0].strip(), 'style_value': style_string.split(':')[1].strip().replace(';', '')}]
	return style_dict


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--timeline', help="Umar Graph timeline file")
	parser.add_argument('--html', help="HTML serialized file")

	args = parser.parse_args()
	html_parser = HTMLParser.HTMLParser()

	#with open(args.timeline, 'r') as f:
	with codecs.open(args.timeline, 'r', encoding="utf-8") as f:
		timeline = json.load(f)
		timeline = timeline['timeline']
		
	
	#with codecs.open(args.html, encoding="iso-8859-1") as f:
	with codecs.open(args.html, 'r', encoding="utf-8") as f:
		#html_content = html_parser.unescape(f.read())
		html_content = f.read()
		html_content = html_content.replace('\xc2\xa0'.decode('utf8'), ' ').replace('\xc3\xa0'.decode('utf8'), ' ')
		soup = BeautifulSoup(html_content, 'html.parser')
		if 'xmlns' in soup.html.attrs:
			del soup.html.attrs['xmlns']
	
	G = build_graph(timeline)
	compare_graph_soup(G, soup)
	#search_graph(G, 'meta', [{'attr_name': 'property','attr_value':"og:url"}, {'attr_name': 'content','attr_value':"https://www.brave.com/"}])
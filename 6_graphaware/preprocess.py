from Neo4jConnector import Neo4jConnector


if __name__ == "__main__":
    ## I think this worked -> didnt do any test here but will incorpoarte one if training fails
    print("Preprocessing graph data in Neo4j...")
    neo4j_connector = Neo4jConnector()
    neo4j_connector.append_ref_node()
    print("Appended reference node with mean features.")
    #neo4j_connector.add_edge_weights_depending_on_time_ratio()
    #print("Added edge weights based on time ratio to previous measurement")
    neo4j_connector.feature_neighborhood_aggregation_function()
    print("Computed neighborhood aggregation features for all nodes.")
    neo4j_connector.close()

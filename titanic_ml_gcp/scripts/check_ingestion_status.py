from google.cloud import aiplatform
from google.cloud.aiplatform_v1.services.featurestore_service import FeaturestoreServiceClient

# Initialize
aiplatform.init(location="us-central1")

# Operation Name
operation_name = "projects/537178553916/locations/us-central1/featurestores/titanic_featurestore/entityTypes/passenger/operations/5262850391040589824"

print(f"Checking operation: {operation_name}")

try:
    # We can use the FeaturestoreServiceClient to get operations if it exposes them, 
    # or use the OperationsClient associated with the transport.
    
    client = FeaturestoreServiceClient()
    # The operations client is usually available via client.transport.operations_client
    # But getting a specific operation by name across services is tricky.
    
    # Better way: Use the generic OperationsClient for the region
    from google.cloud.aiplatform_v1.services.featurestore_service.transports.grpc import FeaturestoreServiceGrpcTransport
    from google.longrunning import operations_pb2
    from google.longrunning import operations_pb2_grpc
    import grpc
    
    # Or simpler, check if FeatureStore object has a method to list operations? No.
    
    # Let's try simpler approach with aiplatform.gapic
    from google.cloud import aiplatform_v1
    
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    ops_client = aiplatform_v1.FeaturestoreServiceClient(client_options=client_options).transport.operations_client
    
    response = ops_client.get_operation(name=operation_name)
    
    print(f"Operation Done: {response.done}")
    
    if response.error.code:
        print(f"Operation Error: {response.error}")
        
    if response.done:
        if response.error.code == 0:
             print("Result: Ingestion Completed Successfully.")
             # print(response) # Response might be empty for void return
        else:
             print("Result: Ingestion Failed.")
    else:
        print("Result: Still Running.")

except Exception as e:
    print(f"Error checking operation: {e}")


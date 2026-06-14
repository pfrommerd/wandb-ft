use graphql_client::GraphQLQuery;

type JSONString = String;
type Int64 = i64;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_run.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct UpsertBucket;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_create_run_files.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct CreateRunFiles;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_create_artifact.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct CreateArtifact;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_create_artifact_files.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct CreateArtifactFiles;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_create_artifact_manifest.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct CreateArtifactManifest;

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "graphql/schema.graphql",
    query_path = "graphql/mutation_commit_artifact.graphql",
    skip_serializing_none,
    response_derives = "Debug"
)]
pub struct CommitArtifact;

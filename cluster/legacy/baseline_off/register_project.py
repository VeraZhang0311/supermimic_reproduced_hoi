"""Create only the user-requested new private W&B project, and verify access."""
from pathlib import Path
import json
import wandb
from wandb.apis._generated import CREATE_PROJECT_GQL, UpsertModelInput

entity='tianzizh-university-of-pennsylvania'
project='SuperMimic_Upstream_Reproduction'
api=wandb.Api()
payload=UpsertModelInput(name=project,entity_name=entity,access='PRIVATE',description='Independent GRASP rerun of Chong Zhang SuperMimic LAFAN1 implementation. Pinned upstream a1508d4 / IsaacLab46d6cc3; original SuperMimic_HOI experiments retained separately.')
api._service_api.execute_graphql(CREATE_PROJECT_GQL,{'input':payload.model_dump()})
query='query VerifyProject($entity: String!, $name: String!) { project(name: $name, entityName: $entity) { name access } }'
response=api._service_api.execute_graphql(query,{'entity':entity,'name':project})
assert response['project']['access'].lower()=='private',response
base=api.settings['base_url'].rstrip('/')
receipt={'entity':entity,'project':project,'access':response['project']['access'],'api_base_url':base,'project_url':wandb.util.app_url(base).rstrip('/')+'/'+entity+'/'+project,'upstream_commit':'a1508d42c9d1699a52e8233ab16d08c488441ab9'}
root=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
(root/'provenance/wandb-project.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

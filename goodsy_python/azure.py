from .setup import *


def paste_dictionary(my_dict, my_str):
    for key,value in my_dict.items():
        my_str = my_str.replace(key,value)
    return (my_str)


def az_list_regions():
    com = 'az account list-locations --output table'
    print(com)
    os.system(com)


def az_list_resourcegroups():
    com = 'az group list -o tsv --query "[*].{name:name}"'
    print(com)
    os.system(com)


def az_create_group(resource_group_name=None, azure_region=None, azure_options=None):
    com = 'az group create --name resource_group_name --location azure_region'
    if azure_options is None:
        if resource_group_name is None or azure_region is None:
            print(com)
        else:
            com = com.replace('resource_group_name', resource_group_name)
            com = com.replace('azure_region', azure_region)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


def az_delete_group(resource_group_name=None, azure_options=None):
    com = 'az group delete -n resource_group_name -y --no-wait'
    if azure_options is None:
        if resource_group_name is None:
            print(com)
        else:
            com = com.replace('resource_group_name', resource_group_name)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create ACR( Azure Container Registry):
def az_create_ACR(unique_acr_name=None, resource_group_name=None,
                                    azure_sku="Standard", azure_options=None):
    com = 'az acr create -n unique_acr_name -g resource_group_name --sku azure_sku --admin-enabled true'
    if azure_options is None:
        my_criteria = [unique_acr_name is None, resource_group_name is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('unique_acr_name', unique_acr_name)
            com = com.replace('resource_group_name', resource_group_name)
            com = com.replace('azure_sku', azure_sku)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create Azure Web App for Containers:
# az appservice plan create -n ptg_test_appserviceplan -g DockerRG --is-linux
def az_create_webapp(appserviceplan_name=None, resource_group_name=None, azure_options=None):
    com = 'az appservice plan create -n appserviceplan_name -g resource_group_name --is-linux'
    if azure_options is None:
        my_criteria = [appserviceplan_name is None, resource_group_name is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('appserviceplan_name', appserviceplan_name)
            com = com.replace('resource_group_name', resource_group_name)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create a custom Docker container Web App:
# To create a web app and configuring it to run a custom Docker container, run the following command:
# az webapp create -n <unique-appname> -g DockerRG -p myappserviceplan -i elnably/dockerimagetest
# az webapp create -n ptg-unique-appname -g DockerRG -p ptg_test_appserviceplan -i elnably/dockerimagetest
def az_create_dockercontainer(unique_appname=None, appserviceplan_name=None, docker_imagename=None, azure_options=None):
    com = 'az webapp create -n ptg-unique-appname -g DockerRG -p appserviceplan_name -i docker_imagename'
    if azure_options is None:
        my_criteria = [appserviceplan_name is None, resource_group_name is None, docker_imagename is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('unique_appname', unique_appname)
            com = com.replace('appserviceplan_name', appserviceplan_name)
            com = com.replace('docker_imagename', docker_imagename)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create an Azure SQL server: Important: Enter a unique SQL server name. Since
# the Azure SQL Server name does not support UPPER / Camel casing naming
# conventions, use lowercase for the DB Server Name field value.
# az sql server create -l <region> -g DockerRG -n <unique-sqlserver-name> -u sqladmin -p P2ssw0rd1234
def az_create_sql_server(unique_sqlserver_name=None, sqlserver_pwd=None,
                resource_group_name=None, azure_region=None, azure_options=None):
    com = 'az sql server create -l azure_region -g resource_group_name -n unique_sqlserver_name -u sqladmin -p sqlserver_pwd'
    if azure_options is None:
        my_criteria = [unique_sqlserver_name is None, sqlserver_pwd is None,
        resource_group_name is None, sqlserver_pwd is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('unique_sqlserver_name', unique_sqlserver_name)
            com = com.replace('sqlserver_pwd', sqlserver_pwd)
            com = com.replace('resource_group_name', resource_group_name)
            com = com.replace('azure_region', azure_region)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create a Database:
#! az sql db create -g DockerRG -s <unique-sqlserver-name> -n mhcdb --service-objective S0
def az_create_sql_database(unique_sqlserver_name=None, resource_group_name=None,
            my_database_name=None, azure_options=None):
    com  = 'az sql db create -g resource_group_name -s unique_sqlserver_name '
    com += '-n my_database_name --service-objective S0'
    if azure_options is None:
        my_criteria = [unique_sqlserver_name is None, resource_group_name is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('my_database_name', my_database_name)
            com = com.replace('unique_sqlserver_name', unique_sqlserver_name)
            com = com.replace('resource_group_name', resource_group_name)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Create a firewall rule for SQL server that allows access from Azure services
# az sql server firewall-rule create --resource-group DockerRG --server
# <your-sqlserver-name> --name AllowAllAzureIps --start-ip-address 0.0.0.0
# --end-ip-address 0.0.0.0
def az_create_firewall_rule(unique_sqlserver_name=None, resource_group_name=None,
                            azure_region=None, azure_options=None):
    com  = ''
    com += 'az sql server firewall-rule create --resource-group resource_group_name '
    com += '--server unique_sqlserver_name --name AllowAllAzureIps '
    com += '--start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0'
    if azure_options is None:
        my_criteria = [unique_sqlserver_name is None,
        resource_group_name is None, sqlserver_pwd is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('unique_sqlserver_name', unique_sqlserver_name)
            com = com.replace('resource_group_name', resource_group_name)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)


# Update web app’s connection string
#az webapp config connection-string set -g DockerRG -n <your-appservice-name> -t
#SQLAzure --settings defaultConnection='Data
#Source=tcp:<your-sqlserver-name>.database.windows.net,1433;Initial
#Catalog=mhcdb;User Id=sqladmin;Password=P2ssw0rd1234;'
def az_update_connection_string(unique_sqlserver_name=None, unique_appname=None,
    resource_group_name=None, azure_region=None, azure_options=None):
    com  = ''
    com += 'az webapp config connection-string set -g resource_group_name '
    com += '-n unique_appname -t SQLAzure --settings defaultConnection='
    com += "'Data Source=tcp:unique_sqlserver_name.database.windows.net,1433;"
    com += "Initial Catalog=mhcdb;User Id=sqladmin;Password=sqlserver_pwd;'"
    if azure_options is None:
        my_criteria = [unique_sqlserver_name is None,
        resource_group_name is None, sqlserver_pwd is None]
        if any(my_criteria):
            print(com)
        else:
            com = com.replace('unique_sqlserver_name', unique_sqlserver_name)
            com = com.replace('resource_group_name', resource_group_name)
            com = com.replace('unique_appname', unique_appname)
            com = com.replace('sqlserver_pwd', sqlserver_pwd)
            print(com)
            os.system(com)
    else:
        com = paste_dictionary(azure_options, com)
        print(com)
        os.system(com)

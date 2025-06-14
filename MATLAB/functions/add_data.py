#!/usr/bin/env python3
# add_data.py
# Adds a new item to a DynamoDB table with a composite key using command-line arguments.
#
# Inputs (via command line arguments):
#   --table             Name of the DynamoDB table (default: "Exponents")
#   --constellationM    Partition key value (required)
#   --simulationId      Sort key value (required)
#   --nodesN            Number of nodes (required)
#   --SNR               Signal-to-noise ratio (required)
#   --transmissionRate  Transmission rate (required)
#   --errorExponent     Error exponent (required)
#   --optimalRho        Optimal rho (required)
#
# Output:
#   Prints a confirmation message after successful insertion.

import argparse
from decimal import Decimal
import boto3

def add_data(table="Exponents", constellationM=None, simulationId=None,
             nodesN=None, SNR=None, transmissionRate=None,
             errorExponent=None, optimalRho=None):
    # Connect to DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table_obj = dynamodb.Table(table)

    # Build item to insert
    item = {
        "constellationM": Decimal(str(constellationM)),
        "simulationId": simulationId,
        "nodesN": Decimal(str(nodesN)),
        "signalNoiseRatio": Decimal(str(SNR)),
        "transmissionRate": Decimal(str(transmissionRate)),
        "errorExponent": Decimal(str(errorExponent)),
        "optimalRho": Decimal(str(optimalRho))
    }

    # Insert item into table
    table_obj.put_item(Item=item)
    print(f"Data inserted into table '{table}' with constellationM = {constellationM} and simulationId = {simulationId}.")

def main():
    parser = argparse.ArgumentParser(description="Add data to a DynamoDB table with a composite key.")
    parser.add_argument("--table", default="Exponents", help="Name of the DynamoDB table.")
    parser.add_argument("--constellationM", type=str, required=True, help="Value for constellationM (partition key).")
    parser.add_argument("--simulationId", type=str, required=True, help="Value for simulationId (sort key).")
    parser.add_argument("--nodesN", type=str, required=True, help="Value for nodesN.")
    parser.add_argument("--SNR", type=str, required=True, help="Value for signalNoiseRatio.")
    parser.add_argument("--transmissionRate", type=str, required=True, help="Value for transmissionRate.")
    parser.add_argument("--errorExponent", type=str, required=True, help="Value for errorExponent.")
    parser.add_argument("--optimalRho", type=str, required=True, help="Value for optimalRho.")
    args = parser.parse_args()

    add_data(
        table=args.table,
        constellationM=args.constellationM,
        simulationId=args.simulationId,
        nodesN=args.nodesN,
        SNR=args.SNR,
        transmissionRate=args.transmissionRate,
        errorExponent=args.errorExponent,
        optimalRho=args.optimalRho
    )

if __name__ == "__main__":
    main()
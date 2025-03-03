#!/usr/bin/env python3
import argparse
from decimal import Decimal
import boto3

def add_data(table="Exponents", constellationM=None, simulationId=None,
             nodesN=None, SNR=None, transmissionRate=None,
             errorExponent=None, optimalRho=None):
    """
    Inserts an item into a DynamoDB table with a composite key.

    Parameters:
        table (str): Name of the DynamoDB table.
        constellationM (str or numeric): Partition key value.
        simulationId (str): Sort key value.
        nodesN (str or numeric): Number of nodes.
        SNR (str or numeric): Signal-to-noise ratio.
        transmissionRate (str or numeric): Transmission rate.
        errorExponent (str or numeric): Error exponent.
        optimalRho (str or numeric): Optimal rho.
        
    Returns:
        None
    """
    # Create a DynamoDB resource. (Make sure your AWS credentials are set.)
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table_obj = dynamodb.Table(table)

    # Construct the item.
    # Convert values to strings (if not already) before wrapping with Decimal.
    item = {
        "constellationM": Decimal(str(constellationM)),
        "simulationId": simulationId,
        "nodesN": Decimal(str(nodesN)),
        "signalNoiseRatio": Decimal(str(SNR)),
        "transmissionRate": Decimal(str(transmissionRate)),
        "errorExponent": Decimal(str(errorExponent)),
        "optimalRho": Decimal(str(optimalRho))
    }

    # Insert the item into the table.
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
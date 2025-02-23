#!/usr/bin/env python3
import argparse
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Attr

def main():
    parser = argparse.ArgumentParser(description="Retrieve precomputed data from a DynamoDB table.")
    parser.add_argument("--table", default="Exponents", help="Name of the DynamoDB table.")
    parser.add_argument("--constellationM", type=str, required=True, help="Value for constellationM (filter).")
    parser.add_argument("--nodesN", type=str, required=True, help="Value for nodesN (filter).")
    parser.add_argument("--SNR", type=str, required=True, help="Value for signalNoiseRatio (filter).")
    parser.add_argument("--transmissionRate", type=str, required=True, help="Value for transmissionRate (filter).")
    args = parser.parse_args()

    # Specify the AWS region (adjust if needed)
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table(args.table)

    # Convert input parameters from strings to Decimal for numeric comparisons
    constellationM_val   = Decimal(args.constellationM)
    nodesN_val           = Decimal(args.nodesN)
    SNR_val              = Decimal(args.SNR)
    transmissionRate_val = Decimal(args.transmissionRate)

    # Build a filter expression using boto3's condition expressions.
    filter_expr = (Attr('constellationM').eq(constellationM_val) & 
                   Attr('nodesN').eq(nodesN_val) &
                   Attr('signalNoiseRatio').eq(SNR_val) &
                   Attr('transmissionRate').eq(transmissionRate_val))
    
    # Execute the scan with the filter expression.
    response = table.scan(FilterExpression=filter_expr)
    items = response.get('Items', [])
    
    if not items:
        print("The result is not found.")
    else:
        # Take the first matching item
        item = items[0]
        errorExponent = item.get('errorExponent', None)
        optimalRho = item.get('optimalRho', None)
        print("Found precomputed result:")
        print("errorExponent:", errorExponent)
        print("optimalRho:", optimalRho)

if __name__ == "__main__":
    main()
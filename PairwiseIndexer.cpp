//
// Created by tzhang on 7/19/24.
//

#include <iostream>
#include "PairwiseIndexer.h"

Indexer::Indexer(unsigned long nItems)
{
	this->nItems = nItems;
	this->itemsToRow = new unsigned long[nItems];
	this->itemsToRow[0] = 0;

	// compute LUT
	unsigned int thisRow;
	for (unsigned long i = 1; i < nItems; i++)
	{
		thisRow = nItems - i;
		this->itemsToRow[i] = this->itemsToRow[i - 1] + thisRow;
	}
}

Indexer::~Indexer()
{
	delete [] this->itemsToRow;
}

unsigned long Indexer::RowIndex(unsigned long long condensed) const
{
	long center;
	long left = 0;
	long right = nItems - 2;
	// binary search for row
	while (left < right)
	{
		center = (left + right) / 2;	// TODO: can change this int division to float division for more speed
		if (itemsToRow[center + 1] <= condensed)
			left = center + 1;
		else
			right = center;
	}
	return left;
}

unsigned long Indexer::ColIndex(unsigned long long condensed, unsigned int row) const
{
	return (row + 1) + (condensed - this->itemsToRow[row]);
}
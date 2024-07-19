//
// Created by tzhang on 7/19/24.
//

#pragma once

/**
 * Indexing with pre-computed LUTs to save smol math ops. To minimize calculations,
 * call RowIndex first, store the value, and call ColIndex with it.
 * The solution from stackoverflow requires a sqrt, a ceil, and a double-to-int cast per condensed index.
 * The indexer only uses comparison and addition/subtraction/division on ints
 */
class Indexer
{
public:
	Indexer(unsigned long n);
	~Indexer();

	/**
	 * Finds the row index corresponding to the condensed index
	 * @param condensed
	 * @return
	 */
	unsigned long RowIndex(unsigned long long condensed) const;

	/**
	 * Finds the column index corresponding to the condensed index.
	 * Needs to be called after RowIndex has been called.
	 * @param condensed
	 * @param row
	 * @return
	 */
	unsigned long ColIndex(unsigned long long condensed, unsigned int row) const;


private:
	unsigned long nItems;
	unsigned long* itemsToRow;					// LUT for number of items up and AND EXCLUDING row
};
package jsat.utils;

import java.util.Arrays;

/**
 * This class provides some useful methods and utilities for implementing Closed
 * Hashing structures
 * 
 * @author Edward Raff
 */
public class ClosedHashingUtil
{
    /**
     * Applying this bitwise AND mask to a long will give the bits corresponding
     * to an integer. 
     */
    public static final int INT_MASK = -1;
    /**
     * This value indicates that that status of an open addressing space is 
     * EMPTY, meaning any value can be stored in it
     */
    public static final byte EMPTY = 0;
    /**
     * This value indicates that the status of an open addressing space is 
     * OCCUPIED, meaning it is in use. 
     */
    public static final byte OCCUPIED = EMPTY + 1;
    /**
     * This value indicates that the status of an open addressing space is 
     * DELETED, meaning i should not stop a search for a value - but the code
     * is free to overwrite the values at this location and change the status 
     * to {@link #OCCUPIED}. <br>
     * This can not be set to {@link #EMPTY} unless we can guarantee that no 
     * value is stored in a index chain that stops at this location. 
     */
    public static final byte DELETED = OCCUPIED + 1;
    
    /**
     * This store the value {@link Integer#MIN_VALUE} in the upper 32 bits of a 
     * long, so that the lower 32 bits can store any regular integer. This 
     * allows a fast way to return 2 integers from one method in the form of a 
     * long. The two values can then be recovered from the upper and lower 32 
     * bits of the long. This is meant to be used as a nonsense default value, 
     * indicating that no information is present in the upper 32 bits. 
     */
    public static final long EXTRA_INDEX_INFO = ((long) Integer.MIN_VALUE) << 32;
    
    /**
     * Gets the next twin prime that is near a power of 2 and greater than or 
     * equal to the given value
     * 
     * @param m the integer to get a twine prime larger than
     * @return the a twin prime greater than or equal to 
     */
    public static int getNextPow2TwinPrime(int m)
    {
        int pos = Arrays.binarySearch(twinPrimesP2, m+1);
        if(pos >= 0)
            return twinPrimesP2[pos];
        else
            return twinPrimesP2[-pos - 1];
    }
    
    /**
     * This array lits twin primes that are just larger than a power of 2. The 
     * prime in the list will be the larger of the twins, so the smaller can be 
     * obtained by subtracting 2 from the value stored. The list is stored in 
     * sorted order.<br>
     * Note, the last value stored is just under 2<sup>31</sup>, where the other 
     * values are just over 2<sup>x</sup> for x &lt; 31
     * 
     */
    public static final int[] twinPrimesP2 = 
    {
        7, //2^2 , twin with 5
        13, //2^3 , twin with 11
        19, //2^4 , twin with 17
        43, //2^5 , twin with 41
        73, //2^6 , twin with 71
        139, //2^7 , twin with 137
        271, //2^8 , twin with 269
        523, //2^9 , twin with 632
        1033, //2^10 , twin with 1031
        2083, //2^11 , twin with 2081
        4129, //2^12 , twin with 4127
        8221, //2^13 , twin with 8219
        16453, //2^14 , twin with 16451
        32803, //2^15 , twin with 32801
        65539, //2^16 , twin with 65537
        131113, //2^17 , twin with 131111
        262153, //2^18 , twin with 262151
        524353, //2^19 , twin with 524351
        1048891, //2^20 , twin with 1048889
        2097259, //2^21 , twin with 2097257
        4194583, //2^22 , twin with 4194581
        8388619, //2^23 , twin with 8388617
        16777291, //2^24 , twin with 16777289
        33554503, //2^25 , twin with 33554501
        67109323, //2^26 , twin with 67109321
        134217781, //2^27 , twin with 134217779
        268435579, //2^28 , twin with 268435577
        536871019, //2^29 , twin with 536871017
        1073741833, //2^30 , twin with 1073741831
        2147482951, //first twin under 2^31, twin with 2147482949
    };
}

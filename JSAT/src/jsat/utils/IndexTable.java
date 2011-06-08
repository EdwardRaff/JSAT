
package jsat.utils;

import java.util.Arrays;

/**
 *
 * @author Edward Raff
 */
public class IndexTable<T extends Comparable<T>>
{
    int n;
    int[] index;

    public IndexTable(double[] array)
    {
        n = array.length;
        index = new int[n];
        for(int i = 0; i < n; i++)
            index[i] = i;
        int start = zerosUpFront(array);
        quickSort(array, start, array.length-1);
        
    }
    
    
    public void swap(int i, int j)
    {
        int tmp = index[i];
        index[i] = index[j];
        index[j] = tmp;
    }
    
    public int index(int i)
    {
        return index[i];
    }
    
    /**
     * Moves all the zeros to the front of the array, and then returns the index
     * of the first non zero item. Assumes all values are >= 0
     * @param numbers
     * @return the index of the first non zero
     */
    private int zerosUpFront(double[] numbers)
    {
        int i = 0;
        for(int k = i; k < numbers.length; k++)
            if(numbers[index(k)] == 0)
                swap(k, i++);
        
        return i;
    }
    
    private void quickSort(double[] numbers, int low, int high)
    {
        int len = high-low+1;
//        if(len  == numbers.length)
//            System.out.println("Hat?");
//        else if(len  == 14)
//            System.out.println("\tBad? 14");
//        else
//            System.out.println("\t" + len);
        
        if(len < 7)
        {
            for(int i = low; i <= high; i++)
            {
                for(int j = i+1; j <= high; j++)
                    if(numbers[index(i)] > numbers[index(j)])
                    {
                        swap(i, j);
                    }
            }
            return;
        }
        
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[index(low + (high - low)/2)]; 

        while (i <= j)
        {
            while (numbers[index(i)] < pivot)
                i++;

            while (numbers[index(j)] > pivot)
                j--;

            if (i <= j)
            {
                swap(i, j);
                i++;
                j--;
            }
        }
        if (low < j)
            quickSort(numbers, low, j);
        if (i < high)
            quickSort(numbers, i, high);
    }
    
}

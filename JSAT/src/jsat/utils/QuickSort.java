
package jsat.utils;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Provides implementations of quicksort. This is for use to obtain explicitly 
 * desired behavior, as well as aovid overhead by explicitly allowing extra 
 * Lists when sorting that are swapped when the array/list being sorted is 
 * swapped. <br>
 * <br>
 * This class exist solely for performance reasons. 
 * 
 * @author Edward Raff
 */
public class QuickSort
{

    private QuickSort()
    {
    }

    private static int med3(double[] x, int a, int b, int c)
    {
        return (x[a] < x[b]
                ? (x[b] < x[c] ? b : x[a] < x[c] ? c : a)
                : (x[b] > x[c] ? b : x[a] > x[c] ? c : a));
    }
    
    private static int med3(float[] x, int a, int b, int c)
    {
        return (x[a] < x[b]
                ? (x[b] < x[c] ? b : x[a] < x[c] ? c : a)
                : (x[b] > x[c] ? b : x[a] > x[c] ? c : a));
    }

    protected static void vecswap(double[] x, int a, int b, int n)
    {
        for (int i = 0; i < n; i++)
            swap(x, a++, b++);
    }

    protected static void vecswap(float[] x, int a, int b, int n)
    {
        for (int i = 0; i < n; i++)
            swap(x, a++, b++);
    }


    private static void vecswap(double[] x, int a, int b, int n, Collection<List<?>> paired)
    {
        for (int i = 0; i < n; i++)
        {
            for (List l : paired)
                Collections.swap(l, a, b);
            swap(x, a++, b++);
        }
    }

    private static void vecswap(float[] x, int a, int b, int n, Collection<List<?>> paired)
    {
        for (int i = 0; i < n; i++)
        {
            for (List l : paired)
                Collections.swap(l, a, b);
            swap(x, a++, b++);
        }
    }

    public static void swap(double[] array, int i, int j)
    {
        double tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
    
    public static void swap(float[] array, int i, int j)
    {
        float tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
    
    /**
     * Conditional swap, only swaps the values if array[i] &gt; array[j]
     * @param array the array to potentially swap values in
     * @param i the 1st index
     * @param j the 2nd index
     */
    public static void swapC(double[] array, int i, int j)
    {
        
        double tmp_i= array[i];
        double tmp_j = array[j];
        if(tmp_i > tmp_j)
        {
            array[i] = tmp_j;
            array[j] = tmp_i;
        }
    }
    
    /**
     * 
     * @param array the array to swap values in
     * @param i the 1st index
     * @param j the 2nd index
     * @param paired a collection of lists, every list will have its indices swapped as well
     */
    public static void swap(double[] array, int i, int j, Collection<List<?>> paired)
    {
        double t = array[i];
        array[i] = array[j];
        array[j] = t;
        for(List l : paired)
            Collections.swap(l, i, j);
    }

        /**
     * 
     * @param array the array to swap values in
     * @param i the 1st index
     * @param j the 2nd index
     * @param paired a collection of lists, every list will have its indices swapped as well
     */
    public static void swap(float[] array, int i, int j, Collection<List<?>> paired)
    {
        float t = array[i];
        array[i] = array[j];
        array[j] = t;
        for(List l : paired)
            Collections.swap(l, i, j);
    }

    /**
     * Performs sorting based on the double values natural comparator. 
     * {@link Double#NaN} values will  not be handled appropriately. 
     * 
     * @param x the array to sort
     * @param start the starting index (inclusive) to sort
     * @param end the ending index (exclusive) to sort
     */
    public static void sort(double[] x, int start, int end)
    {
        int a = start;
        int n = end-start;
        if (n < 7)/* Insertion sort on smallest arrays */
        {
            insertionSort(x, a, end);
            return;
        }
        
        int pm = a + (n/2);  /* Small arrays, middle element */
        if (n > 7)
        {
            int pl = a;
            int pn = a + n - 1;
            if (n > 40) /* Big arrays, pseudomedian of 9 */
            {        
                int s = n / 8;
                pl = med3(x, pl, pl + s, pl + 2 * s);
                pm = med3(x, pm - s, pm, pm + s);
                pn = med3(x, pn - 2 * s, pn - s, pn);
            }
            pm = med3(x, pl, pm, pn); /* Mid-size, med of 3 */
        }
        double pivotValue = x[pm];

        int pa = a, pb = pa, pc = end - 1, pd = pc;
        while (true)
        {
            while (pb <= pc && x[pb] <= pivotValue)
            {
                if (x[pb] == pivotValue)
                    swap(x, pa++, pb);
                pb++;
            }
            while (pc >= pb && x[pc] >= pivotValue)
            {
                if (x[pc] == pivotValue)
                    swap(x, pc, pd--);
                pc--;
            }
            if (pb > pc)
                break;
            swap(x, pb++, pc--);
        }

        
        int s;
        int pn = end;
        s = Math.min(pa - a, pb - pa);
        vecswap(x, a, pb - s, s);
        s = Math.min(pd - pc, pn - pd - 1);
        vecswap(x, pb, pn - s, s);

        //recurse 
        if ((s = pb - pa) > 1)
            sort(x, a, a+s);
        if ((s = pd - pc) > 1)
            sort(x, pn - s, pn);
        
    }

    /**
     * 
     * @param x
     * @param start inclusive
     * @param end exclusive
     */
    public static void insertionSort(double[] x, int start, int end)
    {
        for (int i = start; i < end; i++)
            for (int j = i; j > start && x[j - 1] > x[j]; j--)
                swap(x, j, j - 1);
    }
    
    /**
     * Performs sorting based on the double values natural comparator. 
     * {@link Double#NaN} values will  not be handled appropriately. 
     * 
     * @param x the array to sort
     * @param start the starting index (inclusive) to sort
     * @param end the ending index (exclusive) to sort
     * @param paired a collection of lists, every list will have its indices swapped as well
     */
    public static void sort(double[] x, int start, int end, Collection<List<?>> paired)
    {
        int a = start;
        int n = end-start;
        if (n < 7)/* Insertion sort on smallest arrays */
        {
            for (int i = a; i < end; i++)
                for (int j = i; j > a && x[j - 1] > x[j]; j--)
                    swap(x, j, j - 1, paired);
            return;
        }
        
        int pm = a + (n/2);  /* Small arrays, middle element */
        if (n > 7)
        {
            int pl = a;
            int pn = a + n - 1;
            if (n > 40) /* Big arrays, pseudomedian of 9 */
            {        
                int s = n / 8;
                pl = med3(x, pl, pl + s, pl + 2 * s);
                pm = med3(x, pm - s, pm, pm + s);
                pn = med3(x, pn - 2 * s, pn - s, pn);
            }
            pm = med3(x, pl, pm, pn); /* Mid-size, med of 3 */
        }
        double pivotValue = x[pm];

        int pa = a, pb = pa, pc = end - 1, pd = pc;
        while (true)
        {
            while (pb <= pc && x[pb] <= pivotValue)
            {
                if (x[pb] == pivotValue)
                    swap(x, pa++, pb, paired);
                pb++;
            }
            while (pc >= pb && x[pc] >= pivotValue)
            {
                if (x[pc] == pivotValue)
                    swap(x, pc, pd--, paired);
                pc--;
            }
            if (pb > pc)
                break;
            swap(x, pb++, pc--, paired);
        }

        
        int s;
        int pn = end;
        s = Math.min(pa - a, pb - pa);
        vecswap(x, a, pb - s, s, paired);
        s = Math.min(pd - pc, pn - pd - 1);
        vecswap(x, pb, pn - s, s, paired);

        //recurse 
        if ((s = pb - pa) > 1)
            sort(x, a, a+s, paired);
        if ((s = pd - pc) > 1)
            sort(x, pn - s, pn, paired);
        
    }
    
    /**
     * Performs sorting based on the double values natural comparator.
     * {@link Double#NaN} values will not be handled appropriately.
     *
     * @param x the array to sort
     * @param start the starting index (inclusive) to sort
     * @param end the ending index (exclusive) to sort
     * @param paired a collection of lists, every list will have its indices swapped as well
     */
    public static void sort(float[] x, int start, int end, Collection<List<?>> paired)
    {
        int a = start;
        int n = end-start;
        if (n < 7)/* Insertion sort on smallest arrays */
        {
            for (int i = a; i < end; i++)
                for (int j = i; j > a && x[j - 1] > x[j]; j--)
                    swap(x, j, j - 1, paired);
            return;
        }
        
        int pm = a + (n/2);  /* Small arrays, middle element */
        if (n > 7)
        {
            int pl = a;
            int pn = a + n - 1;
            if (n > 40) /* Big arrays, pseudomedian of 9 */
            {        
                int s = n / 8;
                pl = med3(x, pl, pl + s, pl + 2 * s);
                pm = med3(x, pm - s, pm, pm + s);
                pn = med3(x, pn - 2 * s, pn - s, pn);
            }
            pm = med3(x, pl, pm, pn); /* Mid-size, med of 3 */
        }
        double pivotValue = x[pm];

        int pa = a, pb = pa, pc = end - 1, pd = pc;
        while (true)
        {
            while (pb <= pc && x[pb] <= pivotValue)
            {
                if (x[pb] == pivotValue)
                    swap(x, pa++, pb, paired);
                pb++;
            }
            while (pc >= pb && x[pc] >= pivotValue)
            {
                if (x[pc] == pivotValue)
                    swap(x, pc, pd--, paired);
                pc--;
            }
            if (pb > pc)
                break;
            swap(x, pb++, pc--, paired);
        }

        
        int s;
        int pn = end;
        s = Math.min(pa - a, pb - pa);
        vecswap(x, a, pb - s, s, paired);
        s = Math.min(pd - pc, pn - pd - 1);
        vecswap(x, pb, pn - s, s, paired);

        //recurse 
        if ((s = pb - pa) > 1)
            sort(x, a, a+s, paired);
        if ((s = pd - pc) > 1)
            sort(x, pn - s, pn, paired);
        
    }
}

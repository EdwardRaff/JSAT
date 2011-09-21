
package jsat.utils;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 *
 * @author Edward Raff
 */
public class IndexTable<T extends Comparable<T>>
{
    Integer[] index;

    public IndexTable(double[] array)
    {
        index = new Integer[array.length];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompD(array));
    }
    
    public IndexTable(T[] array)
    {
        index = new Integer[array.length];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompG(array));
    }
    
    public IndexTable(List<T> list)
    {
        index = new Integer[list.size()];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompList(list));
    }
    
    private class IndexViewCompD implements Comparator<Integer> 
    {
        double[] base;

        public IndexViewCompD(double[] base)
        {
            this.base = base;
        }
        
        public int compare(Integer t, Integer t1)
        {
            return Double.compare(base[t], base[t1]);
        }        
    }
    
    private class IndexViewCompG implements Comparator<Integer> 
    {
        T[] base;

        public IndexViewCompG(T[] base)
        {
            this.base = base;
        }

        public int compare(Integer t, Integer t1)
        {
            return base[t].compareTo(base[t1]);
        }        
    }
    
    private class IndexViewCompList implements Comparator<Integer> 
    {
        List<T> base;

        public IndexViewCompList(List<T> base)
        {
            this.base = base;
        }

        public int compare(Integer t, Integer t1)
        {
            return base.get(t).compareTo(base.get(t1));
        }        
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
    
    public int length()
    {
        return index.length;
    }
    
}


package jsat.utils;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Edward Raff
 */
public class ListUtils
{
    /**
     * This method takes a list and breaks it into <tt>chunks<tt> lists backed by the original 
     * list, with elements being equally spaced among the lists. The lists will be returned in
     * order of the consecutive values they represent in the source list.
     * <br><br><b>NOTE</b>: Because the implementation uses {@link List#subList(int, int) }, 
     * changes to the returned lists will be reflected in the source list. 
     * 
     * @param <T>
     * @param source the source list that will be used to back the chunked lists
     * @param count the number of lists to partition the source into. 
     * @return a lists of lists, each of with the same size with at most a difference of 1. 
     */
    public static <T> List<List<T>> splitList(List<T> source, int count)
    {
        if(count <= 0)
            throw new RuntimeException("Chunks must be greater then 0, not " + count);
        List<List<T>> chunks = new ArrayList<List<T>>(count);
        int baseSize = source.size() / count;
        int remainder = source.size() % count;
        int start = 0;
        
        
        for(int i = 0; i < count; i++)
        {
            int end = start+baseSize;
            if(remainder-- > 0)
                end++;
            chunks.add(source.subList(start, end));
            start = end;
        }
        
        return chunks;
    }
}

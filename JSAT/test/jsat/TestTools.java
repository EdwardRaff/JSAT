/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat;

import java.io.*;
import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.utils.IntSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertEquals;

/**
 *
 * @author Edward Raff
 */
public class TestTools
{
    public static void assertEqualsRelDiff(double expected, double actual, double delta)
    {
        double denom = expected;
        if(expected == 0)
            denom = 1e-6;

        double relError = Math.abs(expected-actual)/denom;
        assertEquals(0.0, relError, delta);
    }
    
    /**
     * Creates a deep copy of the given object via serialization. 
     * @param <O> The class of the object
     * @param orig the object to make a copy of
     * @return a copy of the object via serialization
     */
    public static <O extends Object> O deepCopy(O orig)
    {
        Object obj = null;
        try
        {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(bos);
            out.writeObject(orig);
            out.flush();
            out.close();
            
            ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bos.toByteArray()));
            obj = in.readObject();
        }
        catch (IOException e)
        {
            e.printStackTrace();
            throw new RuntimeException("Object couldn't be copied", e);
        }
        catch (ClassNotFoundException e)
        {
            e.printStackTrace();
            throw new RuntimeException("Object couldn't be copied", e);
        }
        return (O) obj;
    }

    /**
     * Evaluates a given clustering by assuming that the true cluster label is in the first categorical feature. Checks to make sure that each cluster is pure in the label
     * @param clusters the clustering to evaluate
     * @return true if the clustering is good, false otherwise
     */
    public static boolean checkClusteringByCat(List<List<DataPoint>> clusters)
    {
        Set<Integer> seenBefore = new IntSet();
        for (List<DataPoint> cluster : clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            if(seenBefore.contains(thisClass) != false)
                return false;
            for (DataPoint dp : cluster)
                if(thisClass != dp.getCategoricalValue(0))
                    return false;
        }
        return true;
    }
}

/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.classifiers.trees;

import java.util.Stack;
import jsat.DataSet;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class ImportanceByUses implements TreeFeatureImportanceInference
{
    private boolean weightByDepth;

    public ImportanceByUses(boolean weightByDepth)
    {
        this.weightByDepth = weightByDepth;
    }

    public ImportanceByUses()
    {
        this(true);
    }
    

    @Override
    public <Type extends DataSet> double[] getImportanceStats(TreeLearner model, DataSet<Type> data)
    {
        double[] features = new double[data.getNumFeatures()];
        
        visit(model.getTreeNodeVisitor(), 0, features);
        
        return features;
    }
    
    private void visit(TreeNodeVisitor node, int curDepth, double[] features )
    {
        if (node == null)//invalid path was added, skip
            return;
        
        for (int feature : node.featuresUsed())
            if (weightByDepth)
                features[feature] += Math.pow(2, -curDepth);
            else
                features[feature]++;

        if (!node.isLeaf())
        {
            for (int i = 0; i < node.childrenCount(); i++)
                visit(node.getChild(i), curDepth + 1, features);
        }
    }
    
}

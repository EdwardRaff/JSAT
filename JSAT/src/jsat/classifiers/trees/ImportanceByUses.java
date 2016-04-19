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

    @Override
    public <Type extends DataSet> double[] getImportanceStats(TreeLearner model, DataSet<Type> data)
    {
        double[] features = new double[data.getNumFeatures()];
        
        Stack<TreeNodeVisitor> nodes = new Stack<TreeNodeVisitor>();
        nodes.add(model.getTreeNodeVisitor());
        
        while(!nodes.isEmpty())//go through and count each feature used!
        {
            
            TreeNodeVisitor node = nodes.pop();
            if(node == null)//invalid path was added, skip
                continue;
            else if(!node.isLeaf())
                for(int i = 0; i < node.childrenCount(); i++)
                    nodes.add(node.getChild(i));
            for(int feature : node.featuresUsed())
                features[feature]++;
        }
           
        
        return features;
    }
    
}

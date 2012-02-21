
package jsat.graphing;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;

/**
 * Plots a given data set and color codes the data points by their class category. 
 * 
 * @author Edward Raff
 */
public class CategoryPlot extends Graph2D
{
    private List<Color> categoryColors;
    private Vec xVals;
    private Vec yVals;
    private int[] category;
    private CategoricalData categories;
    
    public CategoryPlot(ClassificationDataSet dataSet)
    {
        super(0, 1, 0, 1);//We are going to change these soon anyway
        
        if(dataSet.getNumNumericalVars() != 2)
            throw new ArithmeticException("Can not perform scatter plot on " + dataSet.getNumNumericalVars() + " variables");
        
        categories = dataSet.getPredicting();
        
        //Create N visiualy distinct colors 
        categoryColors = new ArrayList<Color>(dataSet.getClassSize());
        float colorFactor = 1.0f/dataSet.getClassSize();
        for(int i = 0; i < dataSet.getClassSize(); i++)
        {
            Color c = Color.getHSBColor(i*colorFactor, 0.5f, 0.5f);
            categoryColors.add(c);
        }
        
        
        
        xVals = dataSet.getNumericColumn(0);
        yVals = dataSet.getNumericColumn(1);
        
        setXMin(xVals.min());
        setXMax(xVals.max());
        setYMin(yVals.min());
        setYMax(yVals.max());
        
        category = new int[dataSet.getSampleSize()];
        for(int i = 0; i < category.length; i++)
            category[i] = dataSet.getDataPointCategory(i);
    }
    
    @Override
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;

        for(int i = 0; i < category.length; i++ )
        {
            g2.setColor(categoryColors.get(category[i]));
            g2.draw(new Ellipse2D.Double(toXCord(xVals.get(i))-3, toYCord(yVals.get(i))-3, 6, 6));
        }
        
        
        //Draw Label Info
        Font font = g2.getFont();
        //First, find longest name to find bounds of the box 
        int width = 0;
        for(int i = 0; i < categoryColors.size(); i++)
            width = Math.max(width, g2.getFontMetrics().stringWidth(categories.getOptionName(i)));
        width += 2 + getPadding();
        
        
        g2.clearRect(getPadding(), getPadding(), width, (font.getSize()+2)*categoryColors.size() + getPadding()/2);
        g2.drawRect(getPadding(), getPadding(), width, (font.getSize()+2)*categoryColors.size() + getPadding()/2);
        
        for(int i = 0; i < categoryColors.size(); i++)
        {
            g2.setColor(categoryColors.get(i));
            g2.drawString(categories.getOptionName(i), 0 + getPadding()*3/2, (i+1)*(font.getSize()+2) + getPadding());
        }

    }
}

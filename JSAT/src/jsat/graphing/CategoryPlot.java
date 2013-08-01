
package jsat.graphing;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
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
    private String[] names;
    private PointShape[] shapes;
    private double pointSize = 6.0;
    private boolean fillPoints = false;
    
    /**
     * Creates a new category plot to visualize the given data set
     * @param dataSet the data set to show
     * @throws ArithmeticException if the data set does not have exactly 2 numerical features. 
     */
    public CategoryPlot(ClassificationDataSet dataSet)
    {
        super(0, 1, 0, 1);//We are going to change these soon anyway
        
        if(dataSet.getNumNumericalVars() != 2)
            throw new ArithmeticException("Can not perform scatter plot on " + dataSet.getNumNumericalVars() + " variables");
        
        CategoricalData categories = dataSet.getPredicting();
        names = new String[categories.getNumOfCategories()];
        for(int i = 0; i < names.length; i++)
            names[i] = categories.getOptionName(i);
        
        //Create N visiualy distinct colors 
        categoryColors = new ArrayList<Color>(dataSet.getClassSize());
        shapes = new PointShape[dataSet.getClassSize()];
        
        float colorFactor = 1.0f/dataSet.getClassSize();
        PointShape[] shapeOptions = ScatterPlot.PointShape.values();
        int curShape = 0;
        for(int i = 0; i < dataSet.getClassSize(); i++)
        {
            Color c = Color.getHSBColor(i*colorFactor, 0.95f, 0.7f);
            categoryColors.add(c);
            shapes[i] = shapeOptions[curShape];
            curShape = (curShape +1) % shapeOptions.length;
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
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        Graphics2D g2 = (Graphics2D)g;
        
        drawPoints(g2, imageWidth, imageHeight, pp);
        
        drawKey(g2, 0, 
                Arrays.asList(names), categoryColors, 
                Arrays.asList(shapes));
    }

    /**
     * Draws the set of points
     * @param g2 the graphics object to draw with 
     * @param width the width of the image
     * @param height the height of the image
     * @param pp the progress panel to indicate with 
     */
    protected void drawPoints(Graphics2D g2, int width, int height, ProgressPanel pp)
    {
        if(pp != null)
        {
            pp.getjProgressBar().setIndeterminate(false);
            pp.getjProgressBar().setMaximum(category.length);
        }
        
        for(int i = 0; i < category.length; i++ )
        {
            g2.setColor(categoryColors.get(category[i]));
            drawPoint(g2, shapes[category[i]], xVals.get(i), yVals.get(i), width, height, pointSize, fillPoints);
            if(pp != null)
                pp.getjProgressBar().setValue(i+1);
        }
    }

    /**
     * Sets the shape used for a specific data point
     * @param i the data point index
     * @param shape the shape to draw
     */
    public void setPointShape(int i, PointShape shape)
    {
        shapes[i] = shape;
    }
    
    /**
     * Returns the color used for the specified category when plotting 
     * @param category the category in question
     * @return the color to be used when drawing
     */
    public Color getCategoryColor(int category)
    {
        return categoryColors.get(category);
    }

    /**
     * Sets the size of data points drawn in pixels
     * @param pointSize the pixel size to draw data points at
     */
    public void setPointSize(double pointSize)
    {
        this.pointSize = pointSize;
    }
    
    /**
     * Returns the pixel size to draw data points at
     * @return pixel size to draw data points at
     */
    public double getPointSize()
    {
        return pointSize;
    }

    /**
     * Sets whether or not data points are drawn as outlines or filled in. 
     * @param fillPoints {@code true} to fill in points, {@code false} to draw outlines. 
     */
    public void setFillPoints(boolean fillPoints)
    {
        this.fillPoints = fillPoints;
    }

    /**
     * Returns whether or not data points are drawn as outlines or filled in. 
     * @return {@code true} to fill in points, {@code false} to draw outlines. 
     */
    public boolean isFillPoints()
    {
        return fillPoints;
    }
    
}

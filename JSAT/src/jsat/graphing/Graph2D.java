

package jsat.graphing;

import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import javax.imageio.ImageIO;
import javax.swing.*;
import jsat.math.Function;

/**
 * Graph2D is a component for drawing using x and y numeric values, that will be
 * automatically converted into the correct coordinates of the image. All 
 * drawing is done in a background thread onto an image, so complicated and long
 * running drawing can be done without locking the UI. Drawing is done by 
 * overriding the 
 * {@link #paintWork(java.awt.Graphics, int, int, jsat.graphing.ProgressPanel) }
 * method. 
 * 
 * @author Edward Raff
 */
public class Graph2D extends JComponent  
{
    protected double xMin;
    protected double xMax;

    protected double yMin;
    protected double yMax;

    protected int PAD;
    protected String xAxisTtile = "X Axis";
    protected String yAxisTtile = "Y Axis";
    protected volatile BufferedImage currentImage;
    
    private boolean forceRedraw = false;
    private boolean drawMarkers = true;

    /**
     * Forces the Graph2D to draw a new bitmap on the next repaint even if the
     * canvas has not changed sizes. 
     */
    protected void forceRedraw()
    {
        this.forceRedraw = true;
    }

    /**
     * Creates a new Graph2D that shows the view of a given range and domain
     * @param xMin the minimum value to show on the x axis
     * @param xMax the maximum value to show on the x axis
     * @param yMin the minimum value to show on the y axis
     * @param yMax the maximum value to show on the y axis
     */
    public Graph2D(double xMin, double xMax, double yMin, double yMax)
    {
        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
        this.PAD = -1;//AUTO SET FOR ME PLEASE
        
        //Add right click menu
        addMouseListener(new MouseAdapter() 
        {
            private void popup(MouseEvent e)
            {
                JPopupMenu popup = getPopupMenu();
                popup.show(e.getComponent(), e.getX(), e.getY());
            }

            @Override
            public void mouseReleased(MouseEvent me)
            {
                if(me.isPopupTrigger())
                    popup(me);
            }

            @Override
            public void mousePressed(MouseEvent me)
            {
                if(me.isPopupTrigger())
                    popup(me);
            }
        });
    }

    /**
     * Controls whether value markers are drawn below the plot graph in the 
     * padding region. 
     * @param drawMarkers <tt>true</tt> to draw value markers, <tt>false</tt> to
     * not draw them. 
     */
    public void setDrawMarkers(boolean drawMarkers)
    {
        this.forceRedraw = this.drawMarkers != drawMarkers;
        this.drawMarkers = drawMarkers;
    }

    /**
     * Returns <tt>true</tt> to if value markers are to be drawn, <tt>false</tt>
     * to if they are not. 
     * @return <tt>true</tt> to if value markers are to be drawn, <tt>false</tt>
     * to if they are not. 
     */
    public boolean hasDrawMarkers()
    {
        return drawMarkers;
    }

    /**
     * Sets the title that is displayed along the Y axis of the graph. 
     * @param yAxisTtile the title to print on the Y axis
     */
    public void setYAxisTtile(String yAxisTtile)
    {
        this.yAxisTtile = yAxisTtile;
    }

    /**
     * Sets the title that is displayed along the X axis of the graph. 
     * @param xAxisTtile the title to print on the X axis
     */
    public void setXAxisTtile(String xAxisTtile)
    {
        this.xAxisTtile = xAxisTtile;
    }

    /**
     * Sets the minimum value that is displayed along the x axis. 
     * @param xMin the new minimum value
     */
    public void setXMin(double xMin)
    {
        this.xMin = xMin;
    }

    /**
     * Sets the maximum value that is displayed along the x axis
     * @param xMax the new maximum value
     */
    public void setXMax(double xMax)
    {
        this.xMax = xMax;
    }

    /**
     * Sets the minimum value that is displayed along the y axis. 
     * @param yMin the new minimum value
     */
    public void setYMin(double yMin)
    {
        this.yMin = yMin;
    }

    /**
     * Sets the maximum value that is displayed along the y axis
     * @param yMax the new maximum value
     */
    public void setYMax(double yMax)
    {
        this.yMax = yMax;
    }

    /**
     * Sets the padding in pixels between the edge of the area used for drawing
     * and the edge of the component itself. 
     * @param p the number of pixels 
     */
    public void setPadding(int p)
    {
        this.PAD = p;
    }

    /**
     * Returns the number of pixels used to separate the edge of the drawing 
     * area and the edges of the component itself. 
     * @return the padding size in pixels
     */
    public int getPadding()
    {
        return PAD;
    }
    
    /**
     * Expands the height or width of the current Graph2D in order to make the 
     * view range a perfect square
     */
    public void toSquareProprotion()
    {
        double w = getWidth()-PAD*2;
        double h = getHeight()-PAD*2;
        
        double xRange = xMax-xMin;
        double yRange = yMax-yMin;
        
        double xRw = xRange/w;
        double yRh = xRange/h;
        
        if(xRange/w < yRange/h)
        {
            double toAdd = (yRh*w-xRange)/2.0;
            xMax += toAdd;
            xMin -= toAdd;
        }
        else if(xRange/w > yRange/h)
        {
            double toAdd = (xRw*h-yRange)/2.0;
            yMax += toAdd;
            yMin -= toAdd;
        }
        
    }
    
    @Override
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        final String drawingMessage = "Rendering Update...";
        FontMetrics fm = g.getFontMetrics();
        
        if(currentImage != null)
        {
            if(currentImage.getWidth() != getWidth() || currentImage.getHeight() != getHeight() || forceRedraw)
            {
                forceRedraw = false;
                if(drawWorker == null)
                {
                    createBackgroundImage();
                }
                
                g.drawImage(currentImage, 0, 0, getWidth(), getHeight(), null);
                g.drawString(drawingMessage, 3, getHeight()-fm.getHeight()/2);
            }
            else
                g.drawImage(currentImage, 0, 0, null);
        }
        else if(currentImage == null)
        {
            if(drawWorker == null)
                createBackgroundImage();
            g.drawString(drawingMessage, getWidth()/2-fm.stringWidth(drawingMessage)/2, getHeight()/2-fm.getHeight()/2);
        }
    }

    /**
     * Creates a new worker to do the image rendering in the background. 
     */
    private void createBackgroundImage()
    {
        drawWorker = new SwingWorker<BufferedImage, Object>() 
        {

            @Override
            protected BufferedImage doInBackground() throws Exception
            {
                BufferedImage newImage = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
                Graphics g = newImage.getGraphics();

                paintWork(g, newImage.getWidth(), newImage.getHeight(), null);

                return newImage;
            }

            @Override
            protected void done()
            {

                SwingUtilities.invokeLater(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        try
                        {
                            currentImage = get();
                            drawWorker = null;
                            revalidate();
                            repaint();
                        }
                        catch (Exception ex)
                        {
                            JOptionPane.showMessageDialog(null, ex.getMessage());
                        }
                    }
                });

            }

            
        };
        
        drawWorker.execute();
    }
    
    protected volatile SwingWorker<BufferedImage, Object> drawWorker;
    
    /**
     * Performs the drawing of the content to be displayed, indicating the width
     * and height that should be used. The width and height may differ from the 
     * value returned by {@link #getWidth() }, since drawing will continue while
     * a resize is in progress. 
     * 
     * @param g the graphics context to draw with
     * @param imageWidth the width to assume for the drawing area
     * @param imageHeight the height to assume for the drawing area
     * @param pp a progress panel, which may be null, to use to indicate the 
     * current progress in drawing.
     */
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        Graphics2D g2 = (Graphics2D)g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);
        int w = imageWidth;
        int h = imageHeight;

        FontMetrics fm = g2.getFontMetrics();

        if(PAD == 0)//No padding, dont draw borders!
            return;
        else if(PAD == -1)//auto set
            PAD = (fm.getHeight()+2)*2;//Auto pad so we have 2 levels

        g2.setColor(Color.black);
         // Draw horizontal.
        g2.draw(new Line2D.Double(PAD, PAD, PAD, h-PAD));
        // Draw vertical.
        g2.draw(new Line2D.Double(PAD, h-PAD, w-PAD, h-PAD));

        
        
        //Draw horizontal marks
        if(drawMarkers)
            for(int x = PAD*2; x <= w-PAD*2; x+= (w-PAD*2)/4)
            {
                double xVal = toXVal(x, imageWidth);
                String xValString = displayValue(xVal);

                g2.drawString(xValString, x-fm.stringWidth(xValString)/2, h-PAD+fm.getHeight());
            }
        
        //draw X axist title
        if(xAxisTtile != null && !xAxisTtile.equals(""))
            g2.drawString(xAxisTtile, (w-PAD)/2-fm.stringWidth(xAxisTtile)/2, h-PAD+fm.getHeight()*2+2);

        //Dray Y axis title verticaly
        AffineTransform at = new AffineTransform();
        at.setToRotation(-Math.PI/2.0, 0, 0);

        Font origFont = getFont();
        Font der = origFont.deriveFont(at);
        g2.setFont(der);

        //Draw vertical marks
        if(drawMarkers)
            for(int y = PAD*2; y <= h-PAD*2; y+= (h-PAD*2)/4)
            {
                double yVal = toYVal(y, imageHeight);
                String yValString = displayValue(yVal);

                g2.drawString(yValString, fm.getHeight()*2, y+fm.getHeight()/2);
            }
        
        if(yAxisTtile != null && !yAxisTtile.equals(""))
            g2.drawString(yAxisTtile, fm.getAscent(), (h+PAD)/2+fm.getHeight());
        
        g2.setFont(origFont);
        
    }
    
    private static final DecimalFormat sigFormat = new DecimalFormat("0.##E0");
    private static String displayValue(double x)
    {
        String tmp = Double.toString(x);
        if(tmp.length() <= 4)
            return tmp;
        else if( x < 99 && x >= -99)
            return tmp.substring(0, 5);
        else if (x > 0 && x < 9999)//If id did not have a decimal poitn in it, we would have returned in the first if
            return tmp.substring(0, 5);
        else if(x < 0 && x > -9999 && tmp.length() > 5)//Same logic
            return tmp.substring(0, 6);
        //Else, sig fig it
        
        return sigFormat.format(x);
    }
    
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the x axis. It assume the current 
     * component width returned by {@link #getWidth() } is correct. 
     * @param x the numeric value to convert to a pixel value
     * @return the integer index that this point would be located at on the x 
     * axis
     */
    public int toXCord(double x)
    {
        return toXCord(x, getWidth());
    }
            
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the x axis. 
     * @param x the numeric value to convert to a pixel value
     * @param width the width of the Graph2D object to use in conversion
     * @return the integer index that this point would be located at on the x 
     * axis using the  specified <tt>width</tt>
     */
    public int toXCord(double x, int width)
    {
        double scale = (width - 2*PAD)/(xMax-xMin);

        x-=xMin;

        return (int)Math.round(x*scale)+PAD;
    }

    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the x axis. It assume the current 
     * component width returned by {@link #getWidth() } is correct. 
     * @param x the numeric value to convert to a pixel value
     * @return the numeric value that would be located at the pixel position 
     * along the x axis. 
     */
    public double toXVal(int x)
    {
        return toXVal(x, getWidth());
    }
    
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the x axis. 
     * @param x the numeric value to convert to a pixel value
     * @param width the width of the Graph2D object to use in conversion
     * @return the numeric value that would be located at the pixel position 
     * along the x axis. 
     */
    public double toXVal(int x, int width)
    {
        x-=PAD;
        double scale = (width - 2*PAD)/(xMax-xMin);

        return x/scale+xMin;
    }
    
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the y axis. It assume the current 
     * component height returned by {@link #getHeight() } is correct. 
     * @param y the numeric value to convert to a pixel value
     * @return the numeric value that would be located at the pixel position 
     * along the x axis.
     */
    public double toYVal(int y)
    {
        return toYVal(getHeight());
    }
    
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the y axis. 
     * @param y the numeric value to convert to a pixel value
     * @param height the height of the Graph2D object to use in conversion
     * @return the numeric value that would be located at the pixel position 
     * along the x axis. 
     */
    public double toYVal(int y, int height)
    {
        y-=PAD;
        y = (height-2*PAD) - y;//Becase Java swing, the y axis is 0 at the top and H at the bottom, oposite of what we want
        double scale = (height - 2*PAD)/(yMax-yMin);

        return y/scale+yMin;
    }
    
    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the y axis. It assume the current 
     * component width returned by {@link #getWidth() } is correct. 
     * @param y the numeric value to convert to a pixel value
     * @return the integer index that this point would be located at on the y 
     * axis using the  specified <tt>height</tt>
     */
    public int toYCord(double y)
    {
        return toYCord(y, getHeight());
    }

    /**
     * This method takes an input numeric value and converts it into the 
     * appropriate pixel index location along the y axis. 
     * @param y the numeric value to convert to a pixel value
     * @param height the height of the Graph2D object to use in conversion
     * @return the integer index that this point would be located at on the y 
     * axis using the  specified <tt>height</tt>
     */
    public int toYCord(double y, int height)
    {
        double scale = (height - 2*PAD)/(yMax-yMin);

        y-=yMin;

        return height - (int)Math.round(y*scale)-PAD;
    }
    
    /**
     * Scales the range of both the x and y axis by the specified value. 
     * @param scale the value to scale the window range by
     * @throws ArithmeticException if the scale provided is not a positive value
     */
    public void scaleCords(double scale)
    {
        if(scale <= 0 || Double.isInfinite(scale) || Double.isNaN(scale))
            throw new ArithmeticException("Invalid scale " + scale);
        double xRange = Math.abs(xMax-xMin);
        double yRange = Math.abs(yMax-yMin);
        
        //Shave, how much to add to each side of the range
        double xShave = (xRange*scale-xRange)/2;
        double yShave = (yRange*scale-yRange)/2;
        
        setXMax(xMax+xShave);
        setXMin(xMin-xShave);
        setYMax(yMax+yShave);
        setYMin(yMin-yShave);
    }

    /**
     * Draws a single variate function using the x axis as the input, and 
     * the response as the y axis value.  
     * 
     * @param func the function to draw
     * @param g the graphics object to draw with
     */
    protected void drawFunction(Graphics2D g, Function func)
    {
        double lastX = toXVal(PAD+1);
        double lastY = func.f(lastX);

        for(int i = PAD+2; i < getWidth()-PAD; i+=4)
        {
            double x = toXVal(i);
            double y = func.f(x);
            
            g.drawLine(toXCord(lastX), toYCord(lastY), toXCord(x), toYCord(y));
            lastX = x;
            lastY = y;
        }
    }

    /**
     * Draws a point object using the specified graphics object, using 
     * floating point values of the pixel coordinates. 
     * 
     * @param pointShape the shape type to draw
     * @param xPos the pixel position of the point on the x axis
     * @param yPos the pixel position of the point on the y axis
     * @param size the diameter of the point. 
     * @param fill <tt>true</tt> to fill the point as a solid color, 
     * <tt>false</tt> to draw a hollow point. 
     * @param g2 the graphics object to draw with 
     */
    protected void drawPoint(Graphics2D g2, PointShape pointShape, double xPos, double yPos, double size, boolean fill)
    {
        Shape shape = null;
        if(pointShape == PointShape.CIRCLE)
            shape = new Ellipse2D.Double(xPos, yPos, size, size);
        else if(pointShape == PointShape.SQUARE)
            shape = new Rectangle2D.Double(xPos, yPos, size, size);
        else if(pointShape == PointShape.TRIANGLE)
        {
            double s = size/2;
            GeneralPath path = new GeneralPath(GeneralPath.WIND_NON_ZERO);
            path.moveTo(s+xPos  , s+yPos-s);
            path.lineTo(s+xPos+s, s+yPos+s);
            path.lineTo(s+xPos-s, s+yPos+s);
            path.closePath();
            shape = path;
        }
        
        
        if(fill)
            g2.fill(shape);
        else 
            g2.draw(shape);
    }

    /**
     * The different supported point shapes that can be drawn
     */
    public enum PointShape 
    {
        CIRCLE, SQUARE, TRIANGLE
    }
    
    /**
     * Draws a point object using the specified graphics object, using 
     * numeric value locations. 
     * 
     * @param g2 the graphics object to draw with 
     * @param xValue the numeric value on the x axis 
     * @param yValue the numeric value on the y axis
     * @param size the diameter of the point to draw
     * @param fill <tt>true</tt> to fill the point as a solid color, 
     * <tt>false</tt> to draw a hollow point. 
     * @param pointShape the shape type to draw
     * @param height the height of the assumed image
     * @param width the width of the assumed image
     */
    protected void drawPoint(Graphics2D g2, PointShape pointShape, double xValue, double yValue, int width, int height, double size, boolean fill)
    {
        double xPos = toXCord(xValue, width)-size/2;
        double yPos = toYCord(yValue, height)-size/2;
        
        drawPoint(g2, pointShape, xPos, yPos, size, fill);
    }

    /**
     * Creates a new popup menu to show when someone right clicks on the component. 
     * @return the popup menu to show when someone right clicks on the component. 
     */
    protected JPopupMenu getPopupMenu()
    {
        JPopupMenu menu = new JPopupMenu("Options");
        
        JMenuItem makeSquareItem = new JMenuItem("Make Proprotional");
        makeSquareItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent ae)
            {
                toSquareProprotion();
                forceRedraw();
                invalidate();
                revalidate();
                repaint();
            }
        });
        menu.add(makeSquareItem);
        
        JMenuItem saveAsImage = new JMenuItem("Render to File");
        saveAsImage.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent ae)
            {
                String input = JOptionPane.showInputDialog(getParent(), "Enter the width and height to use", "600, 600");
                if(input == null || !input.matches("[0-9]+,\\s*[0-9]+"))
                    return;
                String[] splt = input.split(",");
                final int width = Integer.parseInt(splt[0].trim());
                final int height = Integer.parseInt(splt[1].trim());
                
                JFileChooser jfc = new JFileChooser();

                int returnVal = jfc.showSaveDialog(getParent());
                if (returnVal == JFileChooser.APPROVE_OPTION)
                {
                    final File file = jfc.getSelectedFile();
                    final BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
                    final ProgressPanel pp = new ProgressPanel("Rendering Image to file", "");
                    final JDialog dialog = new JDialog();
                    dialog.setContentPane(pp);
                    dialog.pack();
                    dialog.setVisible(true);
                    Thread t = new Thread(new Runnable() {

                        @Override
                        public void run()
                        {
                            paintWork(img.createGraphics(), width, height, pp);
                            dialog.setVisible(false);
                            try
                            {
                                ImageIO.write(img, "png", file);
                            }
                            catch (IOException ex)
                            {
                                SwingUtilities.invokeLater(new Runnable() {

                                    @Override
                                    public void run()
                                    {
                                        JOptionPane.showMessageDialog(getParent(), "An error occured when trying to save the image", "Error", JOptionPane.ERROR_MESSAGE);
                                    }
                                });
                            }
                        }
                    });
                    t.start();
                }
            }
        });
        menu.add(saveAsImage);
        
        
        return menu;
    }
}

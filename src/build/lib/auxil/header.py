import os, os.path, sys, re
from cStringIO import StringIO

class Header(dict):
    """
    Class representing the text fields of an ENVI format header
    """

    def __init__(self):
        
        self['description']="no data"

        self.strings=['description']
        self.lists=['wavelength','fwhm','sigma','band names','default bands',
                    'bbl', 'map info', 'spectra names']
        self.outputorder=['description','samples','lines','bands','header offset','file type','data type','interleave']

    def read(self,fh):
        """parse the header file for the ENVI key/value pairs."""
        key=''
        val=''
        fh = fh.splitlines()

        # Simple finite state machine, where the states are 'nothing'
        # and 'bracket'.  'bracket' indicates that we have encountered
        # an open bracket and reading in a list.  This state continues
        # until we encounter a close bracket.  FIXME: should handle
        # the case where a close bracket appears inside quotes
        state='nothing'
        for txt in fh:
            if state == 'bracket':
                txt=txt.rstrip()
                if re.search('[}]',txt):
                    txt=re.sub('[}]*$','',txt)
                    val+=txt
                    # remove trailing whitespace
                    self[key]=val.rstrip()
                    state='nothing'
                else:
                    val+=txt+'\n'
            else:
                if re.search('=',txt):
                    key,val = [s.strip() for s in re.split('=\s*',txt,1)]
                    key=key.lower()
                    
                    if val[0] == "{":
                        if val[-1] == "}":
                            # single line string: remove braces and spaces
                            self[key] = val[1:-1].strip()
                        else:
                            state='bracket'
                            # Some ENVI header files have opening and closing
                            # braces on different lines, but also have valid
                            # data on the opening line.
                            val = val[1:].strip()
                    else:
                        # remove garbage characters
                        val=re.sub('\{\}[ \r\n\t]*','',val)
                        self[key]=val
        self.fixup()

    def fixup(self):
        """convert any nonstandard keys here..."""
        if 'sigma' in self:
            self['fwhm']=self['sigma']
            del self['sigma']
            
    def str_string(self,key,val):
        return "%s = {%s%s}%s" % (key,'\n',val,'\n')
            
    def __str__(self):
        fs=StringIO()
        fs.write("ENVI"+'\n')
        order=self.keys()
        for key in self.outputorder:
            try:
                i=order.index(key)
                if key in self.lists or key in self.strings: 
                    fs.write(self.str_string(key,self[key]))
                else:
                    fs.write("%s = %s%s" % (key,self[key],'\n'))
                del order[i]
            except ValueError:
                pass
            
        order.sort()
        for key in order:
            val=self[key]
            if key in self.lists or key in self.strings:
                if val:
                    # only write the list if the list has something in
                    # it
                    fs.write(self.str_string(key,val))
            else:
                fs.write("%s = %s%s" % (key,val,'\n'))
        return fs.getvalue()            
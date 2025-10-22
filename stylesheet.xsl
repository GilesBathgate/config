<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:fo="http://www.w3.org/1999/XSL/Format">

  <xsl:template match="/">
    <fo:root>
      <fo:layout-master-set>
        <fo:simple-page-master master-name="A4"
                               page-height="29.7cm"
                               page-width="21cm"
                               margin-top="1cm"
                               margin-bottom="1cm"
                               margin-left="1.5cm"
                               margin-right="1.5cm">
          <fo:region-body margin-top="1.5cm" margin-bottom="1.5cm"/>
          <fo:region-before extent="1cm"/>
          <fo:region-after extent="1cm"/>
        </fo:simple-page-master>
      </fo:layout-master-set>

      <fo:page-sequence master-reference="A4">
        <fo:static-content flow-name="xsl-region-after">
          <fo:block text-align="center" font-size="10pt">
            Page <fo:page-number/>
          </fo:block>
        </fo:static-content>
        <fo:flow flow-name="xsl-region-body">
          <xsl:apply-templates/>
        </fo:flow>
      </fo:page-sequence>
    </fo:root>
  </xsl:template>

  <xsl:template match="document/title">
    <fo:block font-size="24pt" font-weight="bold" text-align="center" space-after="1cm">
      <xsl:value-of select="."/>
    </fo:block>
  </xsl:template>

  <xsl:template match="chapter/title">
    <fo:block font-size="18pt" font-weight="bold" space-before="1cm" space-after="0.5cm">
      <xsl:value-of select="."/>
    </fo:block>
  </xsl:template>

  <xsl:template match="section/title">
    <fo:block font-size="14pt" font-weight="bold" space-before="0.5cm" space-after="0.2cm">
      <xsl:value-of select="."/>
    </fo:block>
  </xsl:template>

  <xsl:template match="p">
    <fo:block font-size="12pt" space-after="0.2cm" text-align="justify">
      <xsl:value-of select="."/>
    </fo:block>
  </xsl:template>

  <xsl:template match="table">
    <fo:table table-layout="fixed" width="100%" space-after="0.5cm">
      <fo:table-body>
        <xsl:apply-templates select="row"/>
      </fo:table-body>
    </fo:table>
  </xsl:template>

  <xsl:template match="row">
    <fo:table-row>
      <xsl:apply-templates select="cell"/>
    </fo:table-row>
  </xsl:template>

  <xsl:template match="cell">
    <fo:table-cell border="1px solid black" padding="5px">
      <xsl:if test="@colspan">
        <xsl:attribute name="number-columns-spanned">
          <xsl:value-of select="@colspan"/>
        </xsl:attribute>
      </xsl:if>
      <fo:block>
        <xsl:value-of select="."/>
      </fo:block>
    </fo:table-cell>
  </xsl:template>

  <xsl:template match="img">
    <fo:block text-align="center" space-after="0.5cm">
      <fo:external-graphic src="url('{@src}')">
        <xsl:attribute name="content-width">
          <xsl:value-of select="(@width div @x-ppi) * 72"/>
          <xsl:text>pt</xsl:text>
        </xsl:attribute>
      </fo:external-graphic>
    </fo:block>
  </xsl:template>

  <xsl:template match="code">
    <fo:block font-family="monospace" font-size="10pt" background-color="#f0f0f0" padding="5px" space-after="0.5cm">
      <xsl:value-of select="."/>
    </fo:block>
  </xsl:template>

</xsl:stylesheet>

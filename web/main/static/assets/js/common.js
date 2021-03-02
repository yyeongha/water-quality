//KOREA

var mobileChk;

//gnb
var gnbInit = function(config){
	this.config = {
		selector : config.selector || ''
	};

	this.init();
	if($(window).width() < 960){
		mobileChk = true;
		this.btnSetOnMobile();
	}else{
		mobileChk = false;
		this.btnSetOnWeb();
	};

	this.resizeHn();
};

gnbInit.prototype = {
	init : function(){
		var owner = this;
		//web
		this.sel = this.config.selector;
		this.depth1 = this.sel.find("> li");
		this.depth2 = this.depth1.find(".depth2 > li");
		this.depth3 = this.depth2.find(".depth3");
		this.depth1H = this.depth1.height();
		this.depth2H = this.depth2.height();
		
		this.offset;
		
		//mobile
		$(".mobile_gnb").on("click", function(e){
			e.preventDefault();

			this.offset = $(document).scrollTop();
			
			$("#fade").show();
			owner.sel.parent().stop().animate({left : 0}, 500);

			$("body").css({
				"overflow" : "hidden",
				"position" : "fixed"
			});

			$("#container").css({
				"position" : "relative",
				"top" : "-" + this.offset + "px"
			});
		});

		this.sel.parent().find(".fade_close").on("click", function(e){
			e.preventDefault();
			
			$("body").removeAttr("style");
			$(document).scrollTop(this.offset);
			$("#container").removeAttr("style");

			owner.sel.parent().stop().animate({left : "-300px"}, 500, function(){
				owner.gnbSet();
			});
			$("#fade").hide();
		});

		//this.resizeHn();
	},

	btnSetOnWeb : function(){
		this.depth1.on("mouseover", function(e){
			$(this).find(".depth2").show();
		});

		this.depth1.on("mouseout", function(e){
			$(this).find(".depth2").hide();
		});

		this.depth2.on("mouseover", function(e){
			$(this).parent().parent().find("> a").addClass("on");
			$(this).find(".depth3").show();
		});

		this.depth2.on("mouseout", function(e){
			if(!$(this).parent().parent().find("> a").hasClass("active")){
				$(this).parent().parent().find("> a").removeClass("on");
			};
			$(this).removeClass("over");
			$(this).find(".depth3").hide();
		});

		this.depth3.on("mouseover", function(e){
			$(this).parent().addClass("over");
		});

		this.depth1.find("> a").off("click");
		this.depth2.off("click");
	},

	btnSetOnMobile : function(){
		var owner = this;
		this.depth1.off("mouseover");
		this.depth1.off("mouseout");
		this.depth2.off("mouseover");
		this.depth2.off("mouseout");
		
		this.depth1.find("> a").on("click", function(e){
			if($(this).parent().find("> ul").hasClass("depth2")){
				e.preventDefault();
				if($(this).parent().find("ul.depth2").css("display") != "none"){
					$(this).parent().find(".depth2").slideUp(300);
					$(this).removeClass("on");

					return;
				};

				owner.mobileGnbSet($(this).parent().index());
			};
		});

		this.depth2.find("> a").on("click", function(e){
			if($(this).parent().hasClass("include") && $(this).parent().find("ul.depth3").css("display") == "none"){
				e.preventDefault();
				owner.mobileGnb2DepthSet($(this).parent().index(), $(this).parent().parent());
			}else if($(this).parent().hasClass("include")){
				e.preventDefault();
				$(this).parent().find(".depth3").slideUp(300);
				$(this).removeClass("on");
			};
		});
	},

	mobileGnbSet : function(num){
		var owner = this;
		this.depth1.each(function(i){
			if(i == num){
				owner.sel.stop().animate({scrollTop : (owner.depth1H * i) + "px"}, 500);
				$(this).find(".depth2").slideDown(300);
				$(this).find(" > a").addClass("on");
			}else{
				$(this).find(".depth2").slideUp(300);
				$(this).find(" > a").removeClass("on");
			};
		});
	},

	mobileGnb2DepthSet : function(num, node){
		var owner = this;
		node.find("> li").each(function(i){
			if(i == num){
				var chkH = (owner.depth1H * $(this).parent().parent().index()) + (owner.depth2H * i);
				owner.sel.stop().animate({scrollTop : chkH + "px"}, 500);
				$(this).find(".depth3").slideDown(300);
				$(this).find(" > a").addClass("on");
			}else{
				$(this).find(".depth3").slideUp(300);
				$(this).find(" > a").removeClass("on");
			};
		});
	},

	gnbSet : function(){
		this.sel.find("ul").hide();
	},

	resizeHn : function(){
		if($(window).width() < 960){
			//mobile
			if(!mobileChk){
				mobileChk = true;
				this.btnSetOnMobile();
			};
			
			if($("img").hasClass("rwd_img")){
				$(".rwd_img").each(function(){
					var imgName = $(this).attr("src");
					imgName = imgName.substr(imgName.length - 3, imgName.length);

					imgCheck($(this), "M", imgName);
				});
			};
			
		}else{
			//web
			if(mobileChk){
				mobileChk = false;
				this.btnSetOnWeb();

				this.sel.parent().css({left : "-300px"});
				this.gnbSet();
				
				$("#fade").hide();

				$("body").removeAttr("style");
				$(document).scrollTop(this.offset);
				$("#container").removeAttr("style");
			};
			
			if($("img").hasClass("rwd_img")){
				if(get_version_IE() <= 8){
					return;
				};

				$(".rwd_img").each(function(){
					var imgName = $(this).attr("src");
					imgName = imgName.substr(imgName.length - 3, imgName.length);

					imgCheck($(this), "W", imgName);
				});
			};
			
		};
		
	}
};

function imgCheck(node, chk, img){
	switch (chk){
		case "W" : 
			if(img == "jpg"){
				node.attr("src", node.attr("src").replace(/(_m.jpg|.jpg)$/i, ".jpg"));
			}else{
				node.attr("src", node.attr("src").replace(/(_m.png|.png)$/i, ".png"));
			};
		break;

		case "M" : 
			if(img == "jpg"){
				node.attr("src", node.attr("src").replace(/(_m.jpg|.jpg)$/i, "_m.jpg"));
			}else{
				node.attr("src", node.attr("src").replace(/(_m.png|.png)$/i, "_m.png"));
			};
		break;
	}
};

//lnb
var lnbInit = function(config){
	this.config = {
		selector : config.selector || ''
	};

	this.init();
};

lnbInit.prototype = {
	init : function(){
		var depth1 = this.config.selector.find("> li");

		depth1.find("> a").on("click", function(e){
			if(!$(this).parent().hasClass("no_depth")){
				e.preventDefault();
				
				if($(this).parent().find("> ul").hasClass("lnb_2depth")){
					lnbSet($(this).parent().index());
				};
			};
		});

		function lnbSet(num){
			depth1.each(function(i){
				if(num == i){
					$(this).addClass("current");
					$(this).find(".lnb_2depth").stop().slideDown(300);
				}else{
					$(this).removeClass("current");
					$(this).find(".lnb_2depth").stop().slideUp(300);
				};
			});
		};
	}
};

//product_info
var productInit = function(config){
	this.config = {
		selector : config.selector || '',
		h : config.h || ''
	};

	this.init();
};

productInit.prototype = {
	init : function(){
		this.h = this.config.h;
		this.sel = this.config.selector;
		this.hArray = [];
		
		//this.sel.find(".product_view").css("height", this.h + "px");
		//$("#content .in_cont > .product_info .product_list > dl").height() + parseInt($("#content .in_cont > .product_info .product_list").css("padding-top")) + parseInt($("#content .in_cont > .product_info .product_list").css("padding-bottom")) + 2

		this.sel.each(function(i){
			var h = $(this).find(".product_list > dl").height() + parseInt($(this).find(".product_list").css("padding-top")) + parseInt($(this).find(".product_list").css("padding-bottom")) + 2
			$(this).find(".product_view").css("height", h + "px");
			//hArray.push(h);
		});
	},

	viewOn : function(node){
		var list = node.parent().find(".product_view");
		this.h = parseInt(node.parent().find(".product_view").css("height"));
		node.css("z-index", "-1");
		
		var chkH = this.viewHeightChk(list);
		
		list.css("visibility", "visible");
		list.stop().animate({"height" : chkH + "px"}, 200);
	},

	viewOff : function(node){
		var firstH = this.h;
		
		node.each(function(i){
			if($(this).find(".product_view").css("visibility") == "visible"){
				$(this).find(".product_view").stop().animate({"height" : firstH + "px"}, 200, function(){
					$(this).css("visibility", "hidden");
					$(this).parent().find(".product_list").css("z-index", 0);
				});
			};
		});
		
	},

	viewHeightChk : function(list){
		var chkH = parseInt(list.css("padding-top")) + parseInt(list.css("padding-bottom")) + list.find("dl").height() + list.find(".detail_info").height() + parseInt(list.find(".detail_info").css("margin-top"));

		return chkH;
	},

	resizeHn : function(){
		this.sel.each(function(){
			if($(this).find(".product_view").css("visibility") == "visible"){
				var chkH = parseInt($(this).find(".product_view").css("padding-top")) + parseInt($(this).find(".product_view").css("padding-bottom")) + $(this).find(".product_view").find("dl").height() + $(this).find(".product_view").find(".detail_info").height() + parseInt($(this).find(".product_view").find(".detail_info").css("margin-top"));
				$(this).find(".product_view").css("height", chkH + "px");
			}else{
				var h = $(this).find(".product_list > dl").height() + parseInt($(this).find(".product_list").css("padding-top")) + parseInt($(this).find(".product_list").css("padding-bottom")) + 2;
				$(this).find(".product_view").css("height", h + "px");
			};
		});
	}
};

var accodionInit = function(){

};

accodionInit.prototype = {
	init : function(node){
		var owner = this;
		node.find("> li > a").on("click", function(e){
			e.preventDefault();

			owner.menuSet($(this).parent().index(), node);
		});
	},

	menuSet : function(num, node){
		node.find("> li").each(function(i){
			if($(this).index() == num){
				if($(this).hasClass("on")){
					$(this).removeClass("on");
				}else{
					$(this).addClass("on");
				};
			}else{
				$(this).removeClass("on");
			};
		});
	}
};



//pop open
function layer_pop_open(){
	$(".layer_pop").show();
	$("#fade").show();

	if($(".layer_pop").height() >= $(window).height()){
		var h = $(window).height() - 20;
		$(".layer_pop").height(h);
		$(".layer_pop").css({"overflow-y" : "auto"});
	};
	//var cw = ($(window).width() - $(".layer_pop").width()) / 2;
	var ch = ($(window).height() - $(".layer_pop").height()) / 2;

	$(".layer_pop").css({top : ch + "px"});
};

function layer_pop_pos(){
	var ch = ($(window).height() - $(".layer_pop").height()) / 2;

	$(".layer_pop").css({top : ch + "px"});
};

//monthly_color slider
function monthly_color_slider(node){
	var slider = node.bxSlider({
		controls : false,
		pager : false,
		onSlideBefore : function(ele, oldIndex, newIndex){
			$(".slide_boxarrow .count > span").text(newIndex + 1);
		}
	});

	$(".slide_boxarrow .count").html("<span>1</span>/" + slider.getSlideCount());

	$(".slide_boxarrow .left").on("click", function(e){
		e.preventDefault();

		slider.goToPrevSlide();
	});

	$(".slide_boxarrow .right").on("click", function(e){
		e.preventDefault();

		slider.goToNextSlide();
	});
};

//branch_month gallery
var conSlider;
var thumbSlider;
var branch_month_slider = function(){};
branch_month_slider.prototype = {
	init : function(conNode, thumbNode, min, max){
		var owner = this;
		this.conNode = conNode;
		this.thumbNode = thumbNode;
		this.min = min;
		this.max = max;
		
		conSlider = this.conNode.bxSlider({
			mode : "fade",
			captions: true,
			controls : false,
			pager : false,
			//infiniteLoop : false,
			onSlideBefore : function(ele, oldIndex, newIndex){
				owner.paging(newIndex + 1, conSlider.getSlideCount());
			}
		});

		$(".pagination .btnPrev").on("click", function(e){
			e.preventDefault();

			conSlider.goToPrevSlide();
		});

		$(".pagination .btnNext").on("click", function(e){
			e.preventDefault();

			conSlider.goToNextSlide();
		});

		if($(window).width() >= 960){
			thumbSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 130,
				slideMargin: 20
			});
		}else{
			thumbSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				minSlides: this.min,
				maxSlides: this.min,
				slideWidth: 304,
				slideMargin: 10
			});
		};

		$(".slideCtrl .btnPrev").on("click", function(e){
			e.preventDefault();

			thumbSlider.goToPrevSlide();
		});

		$(".slideCtrl .btnNext").on("click", function(e){
			e.preventDefault();

			thumbSlider.goToNextSlide();
		});

		var total = conSlider.getSlideCount();
		this.paging(1, total);

		this.thumbNode.find(" > li").on("click", function(e){
			e.preventDefault();

			conSlider.goToSlide($(this).index());
		});

		this.thumbChk = mobileChk;
		
		if(get_version_IE() != 8 && get_version_IE() != 7){
			$(window).resize(function(){
				if(owner.thumbChk == mobileChk){
					return;
				};
				
				owner.thumbChk = mobileChk;
				setTimeout(function(){
					owner.resizeHn();
				}, 10);
			});
		};
		
	},

	paging : function(current, total){
		var pageHtml = "<em>" + current + "</em>/" + total;
		$(".pagination .pageChk").html(pageHtml);

		var thumbPageChk;

		if($(window).width() >= 960){
			thumbPageChk = Math.floor((current - 1) / this.max);
		}else{
			thumbPageChk = Math.floor((current - 1) / this.min);
		};

		thumbSlider.goToSlide(thumbPageChk);

		this.thumbNode.find(" > li").each(function(i){
			if(current - 1 == i){
				$(this).addClass("on");
			}else{
				$(this).removeClass("on");
			};
		});
	},

	resizeHn : function(){
		if(thumbSlider != undefined){
			thumbSlider.destroySlider();
		};
		
		var chkThumb = $(".pagination .pageChk > em").text();
		
		var thumbPageChk;
		
		if($(window).width() >= 960){
			thumbPageChk = Math.floor((chkThumb - 1) / this.max);
			thumbSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				startSlide : thumbPageChk,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 130,
				slideMargin: 20
			});
		}else{
			thumbPageChk = Math.floor((chkThumb - 1) / this.min);
			thumbSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				startSlide : thumbPageChk,
				minSlides: this.min,
				maxSlides: this.min,
				slideWidth: 304,
				slideMargin: 10
			});
		};
	}
};

//color simulation
var colorSlider;
var color_simul = function(){};
color_simul.prototype = {
	init : function(thumbNode, min, max){
		var owner = this;
		this.thumbNode = thumbNode;
		this.min = min;
		this.max = max;
		var total = $(".thumbList .slideBox .frame ul > li").length;
		var current = $(".pagination .pageChk > em").text();
		var thumbPageChk;
		this.paging(current, total);

		if($(window).width() >= 960){
			thumbPageChk = Math.floor(current / this.max);
			colorSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				startSlide : thumbPageChk,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 225,
				slideMargin: 20
			});
		}else{
			thumbPageChk = Math.floor(current / this.min);
			colorSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				startSlide : thumbPageChk,
				minSlides: this.min,
				maxSlides: this.min,
				slideWidth: 300,
				slideMargin: 10
			});
		};

		$(".slideCtrl .btnPrev").on("click", function(e){
			e.preventDefault();

			colorSlider.goToPrevSlide();
		});

		$(".slideCtrl .btnNext").on("click", function(e){
			e.preventDefault();

			colorSlider.goToNextSlide();
		});

		

		this.thumbChk = mobileChk;
		
		if(get_version_IE() != 8 && get_version_IE() != 7){
			$(window).resize(function(){
				if(owner.thumbChk == mobileChk){
					return;
				};
				
				owner.thumbChk = mobileChk;
				setTimeout(function(){
					owner.resizeHn();
				}, 10);
			});
		};
	},
	
	paging : function(current, total){
		//var pageHtml = "<em>" + current + "</em>/" + total;
		//$(".pagination .pageChk").html(pageHtml);

		this.thumbNode.find(" > li").each(function(i){
			if(current - 1 == i){
				$(this).addClass("on");
			}else{
				$(this).removeClass("on");
			};
		});
	},

	resizeHn : function(){
		if(colorSlider != undefined){
			colorSlider.destroySlider();
		};
		
		if($(window).width() >= 960){
			colorSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 225,
				slideMargin: 20
			});
		}else{
			colorSlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				infiniteLoop : false,
				minSlides: this.min,
				maxSlides: this.min,
				slideWidth: 300,
				slideMargin: 10
			});
		};
	}
};

//edu_info gallery
var eduSlider;
var edu_infoGallery = {
	init : function(node){
		var eduSlider = node.find(" > ul").bxSlider({
			controls : false,
			pager : false
		});

		node.find(".btnPrev").on("click", function(e){
			e.preventDefault();

			eduSlider.goToPrevSlide();
		});

		node.find(".btnNext").on("click", function(e){
			e.preventDefault();

			eduSlider.goToNextSlide();
		});
	}
};

//common Gallery
var commonSlider;
var commonGallery = {
	init : function(node){
		this.node = node;
		this.len = node.find(" > ul > li").length;
		this.html = node.find(".playList").html();

		this.dotSet();
		
		if(this.len == 1){
			return;
		};
		
		var owner = this;
		var current = 0;
		var commonSlider = node.find(" > ul").bxSlider({
			controls : false,
			pager : false,
			auto : true,
			pause : 5000,
			onSlideBefore : function(ele, oldIndex, newIndex){
				owner.imgSet(newIndex);
			}
		});

		node.find(".prev").on("click", function(e){
			e.preventDefault();
			commonSlider.goToPrevSlide();
		});

		node.find(".next").on("click", function(e){
			e.preventDefault();
			commonSlider.goToNextSlide();
		});

		node.find(".playList > li").on("click", function(e){
			e.preventDefault();
			commonSlider.goToSlide($(this).index());
		});
	},

	imgSet : function(chk){
		this.node.find(".playList > li").each(function(i){
			if(i == chk){
				$(this).addClass("on");
			}else{
				$(this).removeClass("on");
			};
		});
	},

	dotSet : function(){
		if(this.len == 1){
			this.node.find(".indicator, .left, .right").hide();
			return;
		};

		for(i = 1; i < this.len; i++){
			this.node.find(".playList").append(this.html);
		};

		this.node.find(".playList > li").each(function(i){
			if(i == 0){
				$(this).addClass("on");
			};

			$(this).find(".indent").text($(this).index() + 1);
		});
	}
};

//power colorbook
var power_colorbook = {
	init : function(){
		var owner = this;
		$(".powder_colorbook_list").each(function(){
			$(this).find("li").each(function(i){
				if(i >= 20){
					$(this).hide();
				};
			});
		});

		$(".btn_more").on("click", function(e){
			e.preventDefault();

			var view = owner.viewContents($(this).prev().find("li"));

			owner.moreContents($(this).prev().find("li"), view);
		});
	},

	viewContents : function(node){
		var viewNum;

		node.each(function(i){
			if($(this).css("display") != "none"){
				viewNum = i;
			};
		});

		return viewNum + 20;
	},

	moreContents : function(node, num){
		var total = node.length;
		
		if(num >= total){
			num = total - 1;

			node.parents(".powder_colorbook_list").next().hide();
		};

		node.each(function(i){
			if(i <= num){
				$(this).show();
			};
		});
	}
};

//history
var historySlider;
var historyGallery = {
	init : function(thumbNode, min, max){
		var owner = this;
		this.thumbNode = thumbNode;
		this.min = min;
		this.max = max;

		if($(window).width() >= 960){
			historySlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				//infiniteLoop : false,
				auto : true,
				pause : 5000,
				autoHover : true,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 280,
				slideMargin: 30
			});
		}else{
			historySlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false
				//infiniteLoop : false
			});
		};

		$(".slideCtrl .btnPrev").on("click", function(e){
			e.preventDefault();

			historySlider.goToPrevSlide();
		});

		$(".slideCtrl .btnNext").on("click", function(e){
			e.preventDefault();

			historySlider.goToNextSlide();
		});

		

		this.thumbChk = mobileChk;
		
		if(get_version_IE() != 8 && get_version_IE() != 7){
			$(window).resize(function(){
				if(owner.thumbChk == mobileChk){
					return;
				};
				
				owner.thumbChk = mobileChk;
				setTimeout(function(){
					owner.resizeHn();
				}, 10);
			});
		};
	},

	resizeHn : function(){
		if(historySlider != undefined){
			historySlider.destroySlider();
		};
		
		if($(window).width() >= 960){
			historySlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false,
				//infiniteLoop : false,
				auto : true,
				pause : 5000,
				autoHover : true,
				minSlides: this.max,
				maxSlides: this.max,
				slideWidth: 280,
				slideMargin: 30
			});
		}else{
			historySlider = this.thumbNode.bxSlider({
				controls : false,
				pager : false
				//infiniteLoop : false
			});
		};
	}
};

//input text check
function inputCheck(obj){
	var str = obj.val();
	if(str != "" && str != null && str != undefined){
		obj.addClass("off");
	};

	$(":text").focus(function(){
		$(this).addClass("off");
	}).blur(function(){
		if($(this).val() == "" || $(this).val().length < 1){
			$(this).removeClass("off");
		}
	});
};

//ie version Check
function get_version_IE(){ 
	 var word; 
	 var version = "N/A"; 

	 var agent = navigator.userAgent.toLowerCase(); 
	 var name = navigator.appName; 

	 // IE old version ( IE 10 or Lower ) 
	 if(name == "Microsoft Internet Explorer"){
		 word = "msie ";
	 }else{ 
		 // IE 11 
		 if(agent.search("trident") > -1){
			 word = "trident/.*rv:"; 
		 } else if(agent.search("edge/") > -1){
			 // IE 12  ( Microsoft Edge ) 
			 word = "edge/"; 
		 };
	 };

	 var reg = new RegExp(word + "([0-9]{1,})(\\.{0,}[0-9]{0,1})"); 

	 if (reg.exec(agent) != null) version = RegExp.$1 + RegExp.$2; 

	 return version; 
}  

$(document).ready(function(){
	$(":text").focus(function(){
		$(this).addClass("off");
	}).blur(function(){
		if($(this).val() == "" || $(this).val().length < 1){
			$(this).removeClass("off");
		}
	});
	
	//navigation
	var gnbConfig = {
		selector : $(".gnb_wrap > .depth1")
	};

	var gnb = new gnbInit(gnbConfig);

	var lnbConfig = {
		selector : $("#lnb > .lnb_nav")
	};

	var lnb = new lnbInit(lnbConfig);
	$("#lnb > ul > li.current .lnb_2depth").show();

	//languages
	var foreignCloseH = 0;
	var foreignOpenH = 64;
	$(".foreign > a").on("click", function(e){
		e.preventDefault();
		
		if($(this).parents(".foreign").find(" > ul").height() == 0){
			$(this).addClass("on");
			$(this).parents(".foreign").find(" > ul").stop().animate({height : foreignOpenH + "px"}, 250);
		}else{
			$(this).removeClass("on");
			$(this).parents(".foreign").find(" > ul").stop().animate({height : foreignCloseH + "px"}, 250);
		};
	});

	//search
	$(".search .ic_search").bind("click", function(e){
		e.preventDefault();
		if($(".input_search").css("display") == "none"){
			$(".input_search").stop().slideDown(300);
		}else{
			$(".input_search").stop().slideUp(300);
		};
	});

	$(".sch_close").on("click", function(e){
		e.preventDefault();
		$(".input_search").stop().slideUp(300);
	});

	$(".input_search div.select_category > a").on("click", function(e){
		e.preventDefault();

		var targetH = $(this).parent().find("> ul > li").height();
		var targetL = $(this).parent().find("> ul > li").length;

		$(this).parent().find("> ul").stop().animate({"height" : targetH * targetL + "px"}, 500);
	});

	$(".input_search div.select_category > ul > li").on("click", function(e){
		e.preventDefault();

		$(".input_search div.select_category > ul > li a").removeClass("on");
		$(this).find("a").addClass("on");
		
		$(this).parents(".select_category").find("> a").text($(this).find("a").text());
		$(this).parent().css("height", 0);

	});
	

	//family site
	var familyCloseH = $(".family_site").height();
	var familyOpenH = $(".family_site > ul").height() + familyCloseH + (parseInt($(".family_site > ul").css("padding-top")) * 2);
	$(".family_site > a").on("click", function(e){
		e.preventDefault();

		if($(this).parent().hasClass("on")){
			$(this).parent().removeClass("on");
			$(this).parent().stop().animate({height : familyCloseH + "px"}, 500);
		}else{
			$(this).parent().addClass("on");
			$(this).parent().stop().animate({height : familyOpenH + "px"}, 500);
		};
	});

	//pop close
	$(".layer_pop .layer_close").on("click", function(e){
		e.preventDefault();

		$(".layer_pop").removeAttr("style");
		
		$("#fade").hide();
		$(".layer_pop").hide();
	});

	var productConfig;
	var productInfo;
	
	$(window).load(function(){
		//product_info
		productConfig = {
			selector : $("#content .in_cont > .product_info > ul > li"),
			h : $("#content .in_cont > .product_info .product_list > dl").height() + parseInt($("#content .in_cont > .product_info .product_list").css("padding-top")) + parseInt($("#content .in_cont > .product_info .product_list").css("padding-bottom")) + 2
		};

		if($("#content .in_cont > div").hasClass("product_info")){
			productInfo = new productInit(productConfig);
			$("#content .in_cont > .product_info .product_list").css("cursor", "pointer");

			//+ btn
			$("#content .in_cont > .product_info .product_list").on("click", function(e){
				e.preventDefault();
				
				productInfo.viewOff($("#content .in_cont .product_info > ul > li"));
				
				productInfo.viewOn($(this));
			});

			//- btn
			$("#content .in_cont > .product_info .product_view .more_cont").on("click", function(e){
				e.preventDefault();

				productInfo.viewOff($(this).parents(".product_view").parent());
			});
			
			//x btn
			$("#content .in_cont > .product_info .product_view .pr_detail_function > a.detail_close").on("click", function(e){
				e.preventDefault();

				productInfo.viewOff($(this).parents(".product_view").parent());
			});
		};
	});
	
	
	//resize
	$(window).on("resize", function(){
		/*if($(this).width() < 960){
			$(".search .ic_search").unbind("click");
		}else{
			$(".search .ic_search").bind("click");
		};*/
		gnb.resizeHn();
		
		//colorbook
		if($("body > div").hasClass("colorinfo_layer"))	colorInfo.resizeHn();
		
		//productinfo
		if($("#content .in_cont > div").hasClass("product_info")){
			productInfo.resizeHn();
		};
	});

	$(window).on("scroll", function() {
		if($(window).width() < 960){

		}else{
			if($(".top").hasClass("etc")) return;
			if($(window).scrollTop() + $(window).height() >= $(document).height() - $("#footer_wrap").height()) {
				var targetH = $(window).scrollTop() + $(window).height() - $("#footer_wrap").offset().top + 20;
				$(".top").css("bottom", targetH + "px");
			}else{
				$(".top").css("bottom", "50px");
			};
		};
		
	});
});